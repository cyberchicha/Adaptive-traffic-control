import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from stable_baselines3 import DQN

from config import MODELS_DIR


class TrafficRLAgent:
    """
    Production inference wrapper for the trained DQN traffic signal controller.
    Loads weights from disk, converts structured lane metrics (from CV API or SUMO)
    into the normalized 18-dimensional state vector S_t, and computes optimal
    green phase actions along with Q-values for explainability.
    """

    SUMO_LANES: List[str] = [
        "n2c_1", "n2c_2",
        "s2c_1", "s2c_2",
        "e2c_1", "e2c_2",
        "w2c_1", "w2c_2",
    ]

    # Default mapping from the 6 annotated lanes of highway_mini.mp4 to SUMO approaches
    DEFAULT_CV_TO_SUMO_MAP: Dict[str, str] = {
        "lane_1": "n2c_1",
        "lane_2": "n2c_2",
        "lane_3": "s2c_1",
        "lane_4": "s2c_2",
        "lane_5": "e2c_1",
        "lane_6": "w2c_1",
    }

    ACTION_METADATA: Dict[int, Dict[str, object]] = {
        0: {"name": "Північ–Південь (зелений)", "short": "NS", "sumo_phase": 0},
        1: {"name": "Схід–Захід (зелений)", "short": "EW", "sumo_phase": 3},
    }

    def __init__(
        self,
        model_path=None,
        max_queue_length_m: float = 89.6,
        cv_to_sumo_map: Optional[Dict[str, str]] = None,
    ):
        self.model_path = str(model_path or (MODELS_DIR / "dqn_traffic_agent.zip"))
        self.max_queue_length_m = float(max_queue_length_m)
        self.cv_to_sumo_map = dict(cv_to_sumo_map or self.DEFAULT_CV_TO_SUMO_MAP)
        self.model: Optional[DQN] = None

    def load(self) -> None:
        """
        Loads the trained Stable-Baselines3 DQN model into memory.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained DQN model not found: {self.model_path}")
        self.model = DQN.load(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def build_state_from_metrics(
        self,
        lane_metrics: Dict[str, Dict[str, float]],
        current_green_action: int = 0,
        base_state: Optional[np.ndarray] = None,
        use_cv_mapping: bool = False,
    ) -> np.ndarray:
        """
        Builds a normalized 18-dimensional state vector S_t in [0, 1] from structured
        lane metrics:
          lane_metrics = {
              "<lane_id>": {"queue_length": float (m), "density": float ([0, 1])}
          }
        If `base_state` is provided (hybrid mode), unmapped SUMO lanes retain values
        from `base_state` (e.g., live SUMO readings).
        """
        if base_state is not None and len(base_state) == 18:
            q_norm = list(map(float, base_state[:8]))
            d_norm = list(map(float, base_state[8:16]))
        else:
            q_norm = [0.0] * 8
            d_norm = [0.0] * 8

        lane_to_idx = {lane: idx for idx, lane in enumerate(self.SUMO_LANES)}

        for src_lane, values in lane_metrics.items():
            target_lane = self.cv_to_sumo_map.get(src_lane, src_lane) if use_cv_mapping else src_lane
            idx = lane_to_idx.get(target_lane)
            if idx is None:
                continue

            q_len_m = float(values.get("queue_length", values.get("L", 0.0)))
            density = float(values.get("density", values.get("D", 0.0)))

            q_norm[idx] = float(np.clip(q_len_m / self.max_queue_length_m, 0.0, 1.0))
            d_norm[idx] = float(np.clip(density, 0.0, 1.0))

        phase_one_hot = [1.0, 0.0] if int(current_green_action) == 0 else [0.0, 1.0]
        return np.array(q_norm + d_norm + phase_one_hot, dtype=np.float32)

    @staticmethod
    def format_analyzer_metrics(raw_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Converts TrafficAnalyzer output {"N": {...}, "L": {...}, "D": {...}} into
        the per-lane JSON structure expected by the REST API and coordinator.
        """
        formatted: Dict[str, Dict[str, float]] = {}
        lane_ids = raw_metrics.get("N", {}).keys()
        for lane_id in lane_ids:
            formatted[lane_id] = {
                "queue_count": int(raw_metrics["N"][lane_id]),
                "queue_length": float(raw_metrics["L"][lane_id]),
                "density": float(raw_metrics["D"][lane_id]),
            }
        return formatted

    def _compute_q_values(self, state: np.ndarray) -> Tuple[float, float]:
        """
        Runs a forward pass through the PyTorch Q-network to obtain Q(s, a) values.
        """
        obs_tensor, _ = self.model.policy.obs_to_tensor(state)
        with torch.no_grad():
            q_vals = self.model.q_net(obs_tensor).detach().cpu().numpy().flatten()
        return float(q_vals[0]), float(q_vals[1])

    def predict(
        self,
        state: np.ndarray,
        current_green_action: Optional[int] = None,
        deterministic: bool = True,
    ) -> Dict[str, object]:
        """
        Predicts the optimal target green phase action a_t for state S_t
        and returns explainable decision metadata (Q-values, margin, action label).
        """
        if not self.is_loaded:
            self.load()

        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.shape != (18,):
            raise ValueError(f"Expected 18-dimensional state vector, got shape {state_arr.shape}")

        action_raw, _ = self.model.predict(state_arr, deterministic=deterministic)
        action = int(action_raw)
        q_ns, q_ew = self._compute_q_values(state_arr)

        if current_green_action is None:
            current_green_action = 0 if state_arr[16] >= state_arr[17] else 1

        meta = self.ACTION_METADATA[action]
        will_switch = action != int(current_green_action)
        decision_text = (
            f"Перемкнути на фазу: {meta['name']}"
            if will_switch
            else f"Подовжити фазу: {meta['name']}"
        )

        return {
            "action": action,
            "direction": meta["short"],
            "phase_name": meta["name"],
            "sumo_phase": meta["sumo_phase"],
            "will_switch": will_switch,
            "decision_text": decision_text,
            "q_values": {"NS": round(q_ns, 4), "EW": round(q_ew, 4)},
            "confidence_margin": round(abs(q_ns - q_ew), 4),
        }