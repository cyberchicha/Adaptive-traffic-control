import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import sumolib
import traci

from config import SUMO_CFG_PATH


class TraCIClient:
    """
    Client-server interface to Eclipse SUMO via TraCI API (TCP/IP).
    Implements the 5 core interaction stages specified in Section 3.5:
      1. Connection lifecycle management (connect / close)
      2. Intersection state polling (normalized state vector S_t and telemetry)
      3. Dynamic traffic parameter injection (hybrid/peak-load vehicle spawning)
      4. Adaptive traffic light control with mandatory safety transitions
      5. Blocking step-by-step simulation synchronization
    """

    GREEN_PHASES: Dict[int, int] = {0: 0, 1: 3}
    PED_CLEAR_PHASES: Dict[int, int] = {0: 1, 1: 4}
    YELLOW_PHASES: Dict[int, int] = {0: 2, 1: 5}

    DIRECTION_ROUTES: Dict[str, str] = {
        "north": "route_n2s",
        "south": "route_s2n",
        "west": "route_w2e",
        "east": "route_e2w",
    }

    VEHICLE_LANES: List[str] = [
        "n2c_1", "n2c_2",
        "s2c_1", "s2c_2",
        "e2c_1", "e2c_2",
        "w2c_1", "w2c_2",
    ]

    def __init__(
        self,
        sumo_cfg_path=SUMO_CFG_PATH,
        tls_id: str = "center",
        min_green_time: int = 10,
        ped_clear_time: int = 5,
        yellow_time: int = 3,
        max_sim_time: float = 3600.0,
        halt_speed_threshold: float = 1.38,
        avg_vehicle_length: float = 5.0,
        max_queue_gap_meters: float = 10.0,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        self.sumo_cfg_path = str(sumo_cfg_path)
        self.tls_id = tls_id
        self.min_green_time = int(min_green_time)
        self.ped_clear_time = int(ped_clear_time)
        self.yellow_time = int(yellow_time)
        self.max_sim_time = float(max_sim_time)
        self.halt_speed_threshold = float(halt_speed_threshold)
        self.avg_vehicle_length = float(avg_vehicle_length)
        self.max_queue_gap_meters = float(max_queue_gap_meters)

        default_weights = {
            "queue": 0.4,
            "wait": 0.002,
            "stops": 0.5,
            "throughput": 0.5,
            "pedestrian": 0.04,
        }
        self.reward_weights = {**default_weights, **(reward_weights or {})}

        self._connected = False
        self.render_mode = "none"
        self.control_mode = "dqn"  # "dqn" or "static"
        self.lane_lengths: Dict[str, float] = {}
        self.steps_per_second = 10
        self.current_green_action = 0
        self.green_elapsed = 0.0
        self._prev_incoming_vehs = set()
        self._prev_halted_vehs = set()
        self._injected_counter = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Stage 1: Connection to SUMO
    # ------------------------------------------------------------------
    def connect(
        self,
        render_mode: str = "none",
        control_mode: str = "dqn",
        seed: Optional[int] = 42,
    ) -> None:
        """
        Starts the SUMO server process and establishes a blocking TraCI TCP/IP session.
        """
        self.close()

        self.render_mode = render_mode
        self.control_mode = control_mode
        binary = "sumo-gui" if render_mode == "human" else "sumo"

        sumo_cmd = [
            sumolib.checkBinary(binary),
            "-c", self.sumo_cfg_path,
            "--start",
            "--quit-on-end",
            "--no-step-log",
        ]
        if seed is not None:
            sumo_cmd.extend(["--seed", str(int(seed))])

        traci.start(sumo_cmd)
        self._connected = True

        self.lane_lengths = {lane: float(traci.lane.getLength(lane)) for lane in self.VEHICLE_LANES}
        sim_delta_t = float(traci.simulation.getDeltaT())
        self.steps_per_second = max(1, int(round(1.0 / sim_delta_t)))

        self.current_green_action = 0
        self.green_elapsed = 0.0
        self._prev_incoming_vehs = set()
        self._prev_halted_vehs = set()
        self._injected_counter = 0

        if self.control_mode == "dqn":
            self._set_phase_locked(self.GREEN_PHASES[self.current_green_action])

    def close(self) -> None:
        """
        Safely terminates the active TraCI connection.
        """
        if self._connected:
            try:
                traci.close()
            except traci.exceptions.FatalTraCIError:
                pass
            finally:
                self._connected = False

    # ------------------------------------------------------------------
    # Stage 2: Polling State & Metrics
    # ------------------------------------------------------------------
    def _compute_spatial_queue_length(self, dists_to_stop: List[float]) -> float:
        n = len(dists_to_stop)
        if n == 0:
            return 0.0

        sorted_d = sorted(dists_to_stop)
        total_length = n * self.avg_vehicle_length
        for i in range(n - 1):
            center_dist = sorted_d[i + 1] - sorted_d[i]
            bumper_gap = max(0.0, center_dist - self.avg_vehicle_length)
            if bumper_gap <= self.max_queue_gap_meters:
                total_length += bumper_gap
        return float(total_length)

    def get_state_and_metrics(self) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        Polls all 8 vehicle approach lanes and the traffic light controller,
        returning the normalized 18D state vector S_t and structured telemetry.
        """
        if not self._connected:
            raise RuntimeError("TraCIClient is not connected to SUMO")

        q_norm_list: List[float] = []
        d_norm_list: List[float] = []
        per_lane: Dict[str, Dict[str, float]] = {}

        total_queue_vehicles = 0
        total_queue_length_m = 0.0
        max_queue_length_m = 0.0
        total_waiting_time_s = 0.0
        total_incoming_vehicles = 0

        for lane in self.VEHICLE_LANES:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_length = self.lane_lengths[lane]
            total_incoming_vehicles += len(veh_ids)

            queued_dists = []
            for veh_id in veh_ids:
                total_waiting_time_s += traci.vehicle.getWaitingTime(veh_id)
                if traci.vehicle.getSpeed(veh_id) < self.halt_speed_threshold:
                    pos = traci.vehicle.getLanePosition(veh_id)
                    queued_dists.append(max(0.0, lane_length - pos))

            n_queued = len(queued_dists)
            q_len_m = min(lane_length, self._compute_spatial_queue_length(queued_dists))
            q_norm = min(1.0, max(0.0, q_len_m / lane_length)) if lane_length > 0 else 0.0
            d_norm = min(1.0, max(0.0, float(traci.lane.getLastStepOccupancy(lane))))

            q_norm_list.append(q_norm)
            d_norm_list.append(d_norm)
            per_lane[lane] = {
                "queue_count": int(n_queued),
                "queue_length": round(q_len_m, 2),
                "density": round(d_norm, 4),
            }

            total_queue_vehicles += n_queued
            total_queue_length_m += q_len_m
            if q_len_m > max_queue_length_m:
                max_queue_length_m = q_len_m

        sumo_phase = int(traci.trafficlight.getPhase(self.tls_id))
        active_action = 0 if sumo_phase in (0, 1, 2) else 1
        phase_one_hot = [1.0, 0.0] if active_action == 0 else [0.0, 1.0]
        state = np.array(q_norm_list + d_norm_list + phase_one_hot, dtype=np.float32)

        waiting_pedestrians = sum(
            1 for pid in traci.person.getIDList()
            if traci.person.getWaitingTime(pid) > 0.0
        )

        metrics = {
            "sim_time": round(float(traci.simulation.getTime()), 2),
            "control_mode": self.control_mode,
            "sumo_phase": sumo_phase,
            "active_direction": "NS" if active_action == 0 else "EW",
            "green_elapsed_s": round(self.green_elapsed, 1),
            "total_queue_vehicles": int(total_queue_vehicles),
            "total_queue_length_m": round(total_queue_length_m, 2),
            "max_queue_length_m": round(max_queue_length_m, 2),
            "mean_waiting_time_s": round(
                total_waiting_time_s / total_incoming_vehicles, 2
            ) if total_incoming_vehicles > 0 else 0.0,
            "waiting_pedestrians": int(waiting_pedestrians),
            "lanes": per_lane,
            "is_finished": traci.simulation.getTime() >= self.max_sim_time,
        }
        return state, metrics

    # ------------------------------------------------------------------
    # Stage 3: Dynamic Parameter / Traffic Injection (Hybrid & Peak Mode)
    # ------------------------------------------------------------------
    def inject_vehicles(
        self,
        direction: str,
        count: int = 1,
        veh_type: str = "car",
    ) -> List[str]:
        """
        Dynamically spawns `count` vehicles on the specified approach route
        ('north', 'south', 'east', 'west') to reproduce peak loads or CV-detected surges.
        """
        if not self._connected:
            raise RuntimeError("TraCIClient is not connected to SUMO")

        route_id = self.DIRECTION_ROUTES.get(direction.lower())
        if route_id is None:
            raise ValueError(
                f"Unknown direction '{direction}'. Use one of {list(self.DIRECTION_ROUTES.keys())}"
            )

        spawned_ids: List[str] = []
        for _ in range(max(0, int(count))):
            self._injected_counter += 1
            veh_id = f"ext_{direction.lower()}_{self._injected_counter}"
            traci.vehicle.add(vehID=veh_id, routeID=route_id, typeID=veh_type, departLane="best")
            spawned_ids.append(veh_id)

        return spawned_ids

    # ------------------------------------------------------------------
    # Stages 4 & 5: Traffic Light Control & Blocking Step Synchronization
    # ------------------------------------------------------------------
    def _set_phase_locked(self, phase_index: int) -> None:
        traci.trafficlight.setPhase(self.tls_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000.0)

    def _compute_reward(self, step_acc: Dict[str, float]) -> float:
        w = self.reward_weights
        return float(
            -w["queue"] * step_acc["sum_q_norm_s"]
            - w["wait"] * step_acc["wait_time_s"]
            - w["stops"] * step_acc["new_stops"]
            + w["throughput"] * step_acc["throughput"]
            - w["pedestrian"] * step_acc["wait_peds_s"]
        )

    def advance_simulation(self, seconds: int) -> Dict[str, float]:
        """
        Advances SUMO synchronously for `seconds` simulated seconds, collecting
        per-second queues, waiting times, stops, throughput, and pedestrian metrics.
        """
        if not self._connected:
            raise RuntimeError("TraCIClient is not connected to SUMO")

        acc = {
            "seconds": 0,
            "new_stops": 0,
            "throughput": 0,
            "sum_q_norm_s": 0.0,
            "wait_time_s": 0.0,
            "wait_peds_s": 0,
            "queue_veh_s": 0,
            "queue_len_s": 0.0,
            "mean_wait_s_sum": 0.0,
            "max_queue_len_m": 0.0,
        }

        for _ in range(int(seconds)):
            if traci.simulation.getTime() >= self.max_sim_time:
                break

            for _ in range(self.steps_per_second):
                traci.simulationStep()
                if self.render_mode == "human":
                    time.sleep(0.015)
                if traci.simulation.getTime() >= self.max_sim_time:
                    break

            acc["seconds"] += 1

            curr_incoming = set()
            curr_halted = set()
            sec_q_norm = 0.0
            sec_wait = 0.0
            sec_q_len = 0.0
            sec_max_q = 0.0

            for lane in self.VEHICLE_LANES:
                lane_length = self.lane_lengths[lane]
                queued_dists = []

                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    curr_incoming.add(veh_id)
                    sec_wait += traci.vehicle.getWaitingTime(veh_id)
                    if traci.vehicle.getSpeed(veh_id) < self.halt_speed_threshold:
                        curr_halted.add(veh_id)
                        pos = traci.vehicle.getLanePosition(veh_id)
                        queued_dists.append(max(0.0, lane_length - pos))

                q_len = min(lane_length, self._compute_spatial_queue_length(queued_dists))
                sec_q_len += q_len
                if q_len > sec_max_q:
                    sec_max_q = q_len
                if lane_length > 0:
                    sec_q_norm += min(1.0, max(0.0, q_len / lane_length))

            sec_peds = sum(
                1 for pid in traci.person.getIDList()
                if traci.person.getWaitingTime(pid) > 0.0
            )
            sec_mean_wait = (sec_wait / len(curr_incoming)) if curr_incoming else 0.0

            acc["throughput"] += len(self._prev_incoming_vehs - curr_incoming)
            acc["new_stops"] += len(curr_halted - self._prev_halted_vehs)
            acc["sum_q_norm_s"] += sec_q_norm
            acc["wait_time_s"] += sec_wait
            acc["wait_peds_s"] += sec_peds
            acc["queue_veh_s"] += len(curr_halted)
            acc["queue_len_s"] += sec_q_len
            acc["mean_wait_s_sum"] += sec_mean_wait
            if sec_max_q > acc["max_queue_len_m"]:
                acc["max_queue_len_m"] = sec_max_q

            self._prev_incoming_vehs = curr_incoming
            self._prev_halted_vehs = curr_halted

        return acc

    def apply_action(self, target_action: int, delta_time: int = 5) -> Dict[str, object]:
        """
        Applies the DQN agent's target green phase (0: NS, 1: EW) with safety
        enforcement (min_green_time, pedestrian clearance, and 3 s yellow phase),
        then advances simulation by `delta_time` seconds.
        """
        if not self._connected:
            raise RuntimeError("TraCIClient is not connected to SUMO")

        target_action = int(target_action)
        switched = False

        step_acc = {
            "seconds": 0,
            "new_stops": 0,
            "throughput": 0,
            "sum_q_norm_s": 0.0,
            "wait_time_s": 0.0,
            "wait_peds_s": 0,
            "queue_veh_s": 0,
            "queue_len_s": 0.0,
            "mean_wait_s_sum": 0.0,
            "max_queue_len_m": 0.0,
        }

        def _merge(part: Dict[str, float]) -> None:
            for k in step_acc:
                if k == "max_queue_len_m":
                    step_acc[k] = max(step_acc[k], part[k])
                else:
                    step_acc[k] += part[k]

        if target_action != self.current_green_action and self.green_elapsed >= self.min_green_time:
            switched = True
            if self.ped_clear_time > 0:
                self._set_phase_locked(self.PED_CLEAR_PHASES[self.current_green_action])
                _merge(self.advance_simulation(self.ped_clear_time))

            if self.yellow_time > 0 and traci.simulation.getTime() < self.max_sim_time:
                self._set_phase_locked(self.YELLOW_PHASES[self.current_green_action])
                _merge(self.advance_simulation(self.yellow_time))

            self.current_green_action = target_action
            self.green_elapsed = 0.0
            self._set_phase_locked(self.GREEN_PHASES[self.current_green_action])

        if traci.simulation.getTime() < self.max_sim_time:
            green_part = self.advance_simulation(delta_time)
            self.green_elapsed += green_part["seconds"]
            _merge(green_part)

        state, metrics = self.get_state_and_metrics()
        secs = max(1, step_acc["seconds"])
        reward = self._compute_reward(step_acc)

        return {
            "state": state,
            "metrics": metrics,
            "reward": round(reward, 4),
            "phase_switched": switched,
            "step_seconds": int(step_acc["seconds"]),
            "step_avg_queue_vehicles": round(step_acc["queue_veh_s"] / secs, 2),
            "step_avg_queue_length_m": round(step_acc["queue_len_s"] / secs, 2),
            "step_max_queue_length_m": round(step_acc["max_queue_len_m"], 2),
            "step_mean_waiting_time_s": round(step_acc["mean_wait_s_sum"] / secs, 2),
            "step_avg_waiting_pedestrians": round(step_acc["wait_peds_s"] / secs, 2),
            "new_stops": int(step_acc["new_stops"]),
            "throughput": int(step_acc["throughput"]),
        }

    def step_static(self, delta_time: int = 5) -> Dict[str, object]:
        """
        Advances the simulation by `delta_time` seconds under SUMO's native
        fixed-time traffic light controller.
        """
        step_acc = self.advance_simulation(delta_time)
        state, metrics = self.get_state_and_metrics()
        secs = max(1, step_acc["seconds"])
        reward = self._compute_reward(step_acc)

        return {
            "state": state,
            "metrics": metrics,
            "reward": round(reward, 4),
            "phase_switched": False,
            "step_seconds": int(step_acc["seconds"]),
            "step_avg_queue_vehicles": round(step_acc["queue_veh_s"] / secs, 2),
            "step_avg_queue_length_m": round(step_acc["queue_len_s"] / secs, 2),
            "step_max_queue_length_m": round(step_acc["max_queue_len_m"], 2),
            "step_mean_waiting_time_s": round(step_acc["mean_wait_s_sum"] / secs, 2),
            "step_avg_waiting_pedestrians": round(step_acc["wait_peds_s"] / secs, 2),
            "new_stops": int(step_acc["new_stops"]),
            "throughput": int(step_acc["throughput"]),
        }