"""
Integration test for Stage 1: TraCIClient (SUMO TCP/IP interface)
and TrafficRLAgent (production DQN inference & CV-to-SUMO state mapping).
"""

import argparse
import logging
import os
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from config import MODELS_DIR, SUMO_CFG_PATH
from core.rl.agent import TrafficRLAgent
from core.simulation.traci_client import TraCIClient


def test_agent_cv_mapping_and_inference(agent: TrafficRLAgent) -> None:
    """
    Verifies conversion of CV JSON metrics into a normalized 18D state vector
    and explainable DQN inference with Q-values.
    """
    logging.info("--- Test 1: TrafficRLAgent CV-to-SUMO mapping & inference ---")

    mock_cv_metrics = {
        "lane_1": {"queue_count": 3, "queue_length": 18.5, "density": 0.42},
        "lane_2": {"queue_count": 2, "queue_length": 12.0, "density": 0.28},
        "lane_3": {"queue_count": 1, "queue_length": 5.0, "density": 0.15},
        "lane_4": {"queue_count": 0, "queue_length": 0.0, "density": 0.05},
        "lane_5": {"queue_count": 4, "queue_length": 25.0, "density": 0.65},
        "lane_6": {"queue_count": 0, "queue_length": 0.0, "density": 0.08},
    }

    state = agent.build_state_from_metrics(
        lane_metrics=mock_cv_metrics,
        current_green_action=0,
        use_cv_mapping=True,
    )

    assert state.shape == (18,), f"Expected shape (18,), got {state.shape}"
    assert state.dtype == np.float32, f"Expected float32, got {state.dtype}"
    assert np.all(state >= 0.0) and np.all(state <= 1.0), "State values out of [0, 1] range"
    assert state[16] == 1.0 and state[17] == 0.0, "Invalid one-hot phase encoding for action 0"

    decision = agent.predict(state, current_green_action=0)
    assert decision["action"] in (0, 1), f"Invalid action: {decision['action']}"
    assert "NS" in decision["q_values"] and "EW" in decision["q_values"]

    logging.info(
        f"Mock CV inference OK | {decision['decision_text']} | "
        f"Q={decision['q_values']} | margin={decision['confidence_margin']}"
    )


def test_traci_dqn_loop(agent: TrafficRLAgent, render_mode: str = "none", steps: int = 8) -> None:
    """
    Verifies all 5 TraCI interaction stages (connect, poll, inject, control, sync)
    coupled with TrafficRLAgent inference.
    """
    logging.info("--- Test 2: TraCIClient + TrafficRLAgent closed-loop (DQN mode) ---")

    client = TraCIClient(sumo_cfg_path=SUMO_CFG_PATH)
    try:
        # Stage 1: Connect
        client.connect(render_mode=render_mode, control_mode="dqn", seed=42)
        assert client.is_connected, "TraCIClient failed to report connected status"

        # Stage 2: Initial state polling
        state, metrics = client.get_state_and_metrics()
        assert state.shape == (18,), f"Expected 18D state, got {state.shape}"
        assert len(metrics["lanes"]) == 8, f"Expected 8 vehicle lanes, got {len(metrics['lanes'])}"

        # Stage 3: Dynamic vehicle injection (peak/hybrid scenario)
        spawned = client.inject_vehicles("north", count=3)
        assert len(spawned) == 3, f"Expected 3 injected vehicles, got {len(spawned)}"
        logging.info(f"Injected vehicles on north approach: {spawned}")

        # Stages 4 & 5: Closed-loop control with safety transitions & blocking sync
        for step in range(1, steps + 1):
            decision = agent.predict(state, current_green_action=client.current_green_action)
            result = client.apply_action(decision["action"], delta_time=5)
            state = result["state"]
            m = result["metrics"]

            assert state.shape == (18,)
            assert np.all(state >= 0.0) and np.all(state <= 1.0)

            logging.info(
                f"Step {step:02d} | t={m['sim_time']:5.1f}s | dir={m['active_direction']} | "
                f"switched={str(result['phase_switched']):5s} | {decision['decision_text']} | "
                f"Q_veh={result['step_avg_queue_vehicles']:4.1f} | "
                f"Thr={result['throughput']:2d} | R={result['reward']:6.2f}"
            )

        # Verify hybrid fusion: override mapped lanes from CV onto live SUMO state
        hybrid_state = agent.build_state_from_metrics(
            lane_metrics={"lane_5": {"queue_length": 30.0, "density": 0.75}},
            current_green_action=client.current_green_action,
            base_state=state,
            use_cv_mapping=True,
        )
        assert hybrid_state.shape == (18,)
        assert np.isclose(hybrid_state[12], 0.75, atol=1e-4), "Hybrid CV density override failed for e2c_1"
        logging.info("Hybrid CV + SUMO state fusion verified successfully.")

    finally:
        client.close()
        assert not client.is_connected, "TraCIClient did not close cleanly"


def test_traci_static_mode(render_mode: str = "none", steps: int = 3) -> None:
    """
    Verifies TraCIClient operation in native SUMO static traffic light mode.
    """
    logging.info("--- Test 3: TraCIClient native static mode ---")

    client = TraCIClient(sumo_cfg_path=SUMO_CFG_PATH)
    try:
        client.connect(render_mode=render_mode, control_mode="static", seed=42)
        for step in range(1, steps + 1):
            result = client.step_static(delta_time=5)
            m = result["metrics"]
            assert result["state"].shape == (18,)
            logging.info(
                f"Static Step {step:02d} | t={m['sim_time']:5.1f}s | "
                f"phase={m['sumo_phase']} | Q_veh={result['step_avg_queue_vehicles']:4.1f} | "
                f"R={result['reward']:6.2f}"
            )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration test for TraCIClient and TrafficRLAgent")
    parser.add_argument("--render", action="store_true", help="Launch SUMO with GUI")
    args = parser.parse_args()

    render_mode = "human" if args.render else "none"
    model_path = MODELS_DIR / "dqn_traffic_agent.zip"

    if not os.path.exists(SUMO_CFG_PATH):
        logging.error(f"SUMO config not found: {SUMO_CFG_PATH}")
        return
    if not os.path.exists(model_path):
        logging.error(f"Trained DQN model not found: {model_path}")
        return

    agent = TrafficRLAgent(model_path=model_path)
    agent.load()
    assert agent.is_loaded, "Agent failed to load DQN model"

    test_agent_cv_mapping_and_inference(agent)
    test_traci_dqn_loop(agent, render_mode=render_mode, steps=8)
    test_traci_static_mode(render_mode=render_mode, steps=3)

    logging.info("ALL STAGE 1 TESTS PASSED!")


if __name__ == "__main__":
    main()