import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib


class TrafficEnv(gym.Env):
    """
    Gymnasium environment for Adaptive Traffic Signal Control in Eclipse SUMO.

    State S_t (18 normalized features in [0, 1]):
      - 8 normalized spatial queue lengths q_i (one per vehicle lane)
      - 8 normalized lane occupancy ratios d_i (one per vehicle lane)
      - 2 one-hot flags encoding the active green phase (NS vs EW)

    Action A_t (Discrete(2) - direct target green phase selection):
      - 0: North-South green (SUMO phase 0)
      - 1: East-West green (SUMO phase 3)
      Switching triggers mandatory pedestrian clearance (5 s) and yellow (3 s) phases.
    """

    metadata = {"render_modes": ["human", "none"]}

    # Mapping from logical action index (0: NS, 1: EW) to SUMO tlLogic phase indices
    GREEN_PHASES = {0: 0, 1: 3}
    PED_CLEAR_PHASES = {0: 1, 1: 4}
    YELLOW_PHASES = {0: 2, 1: 5}

    def __init__(
        self,
        sumo_cfg_path,
        render_mode: str = "none",
        tls_id: str = "center",
        delta_time: int = 5,
        min_green_time: int = 10,
        ped_clear_time: int = 5,
        yellow_time: int = 3,
        max_sim_time: float = 3600.0,
        halt_speed_threshold: float = 1.38,  # 5 km/h in m/s
        avg_vehicle_length: float = 5.0,
        max_queue_gap_meters: float = 10.0,
        reward_weights: dict = None,
    ):
        super().__init__()
        self.sumo_cfg_path = str(sumo_cfg_path)
        self.render_mode = render_mode
        self.tls_id = tls_id

        self.delta_time = int(delta_time)
        self.min_green_time = int(min_green_time)
        self.ped_clear_time = int(ped_clear_time)
        self.yellow_time = int(yellow_time)
        self.max_sim_time = float(max_sim_time)

        self.halt_speed_threshold = float(halt_speed_threshold)
        self.avg_vehicle_length = float(avg_vehicle_length)
        self.max_queue_gap_meters = float(max_queue_gap_meters)

        # Reward weights aligned with thesis Section 2.3.3 (+ pedestrian waiting priority)
        default_weights = {
            "queue": 0.4,       # w1: penalty for normalized spatial queues
            "wait": 0.002,       # w2: penalty for accumulated vehicle waiting time (s)
            "stops": 0.5,       # w3: penalty for new vehicle stops
            "throughput": 0.5,  # w4: reward for vehicles clearing the intersection
            "pedestrian": 0.04,  # pedestrian waiting penalty
        }
        self.reward_weights = {**default_weights, **(reward_weights or {})}

        # 8 vehicle approach lanes only (excluding pedestrian sidewalks *_0)
        self.lanes = [
            "n2c_1", "n2c_2",
            "s2c_1", "s2c_2",
            "e2c_1", "e2c_2",
            "w2c_1", "w2c_2",
        ]
        self.num_lanes = len(self.lanes)

        # Runtime state attributes (initialized cleanly before reset)
        self.lane_lengths = {}
        self.steps_per_second = 10
        self.step_counter = 0
        self.current_green_action = 0
        self.green_elapsed = 0.0
        self._prev_incoming_vehs = set()
        self._prev_halted_vehs = set()

        # Action space: direct choice of target green phase (0: NS, 1: EW)
        self.action_space = spaces.Discrete(2)

        # Observation space: 8 q_i + 8 d_i + 2 one-hot phase = 18 features in [0.0, 1.0]
        obs_shape = (self.num_lanes * 2) + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_shape,), dtype=np.float32
        )

    def _compute_spatial_queue_length(self, dists_to_stop: list) -> float:
        """
        Computes spatial queue length (meters) as the sum of halted vehicle lengths
        and inter-vehicle gaps not exceeding max_queue_gap_meters.
        """
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

    def _set_phase_locked(self, phase_index: int) -> None:
        """
        Activates a SUMO traffic light phase and disables automatic static switching.
        """
        traci.trafficlight.setPhase(self.tls_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tls_id, 10000.0)

    def _advance_simulation(self, seconds: int) -> dict:
        """
        Advances SUMO by `seconds` simulated seconds, integrating both
        event-based (stops, throughput) and time-based (queues, waiting, pedestrians)
        reward components every simulated second.
        """
        acc = {
            "seconds": 0,
            "new_stops": 0,
            "throughput": 0,
            "sum_q_norm_s": 0.0,
            "wait_time_s": 0.0,
            "wait_peds_s": 0,
        }

        for _ in range(int(seconds)):
            if traci.simulation.getTime() >= self.max_sim_time:
                break

            for _ in range(self.steps_per_second):
                traci.simulationStep()
                if self.render_mode == "human":
                    time.sleep(0.02)
                if traci.simulation.getTime() >= self.max_sim_time:
                    break

            acc["seconds"] += 1

            curr_incoming = set()
            curr_halted = set()
            sec_q_norm = 0.0
            sec_wait = 0.0

            for lane in self.lanes:
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
                if lane_length > 0:
                    sec_q_norm += min(1.0, max(0.0, q_len / lane_length))

            acc["throughput"] += len(self._prev_incoming_vehs - curr_incoming)
            acc["new_stops"] += len(curr_halted - self._prev_halted_vehs)
            acc["sum_q_norm_s"] += sec_q_norm
            acc["wait_time_s"] += sec_wait
            acc["wait_peds_s"] += sum(
                1 for pid in traci.person.getIDList()
                if traci.person.getWaitingTime(pid) > 0.0
            )

            self._prev_incoming_vehs = curr_incoming
            self._prev_halted_vehs = curr_halted

        return acc

    def _get_state_and_info(self, new_stops: int = 0, throughput: int = 0) -> tuple:
        """
        Reads per-lane normalized spatial queue lengths q_i and occupancies d_i
        from SUMO and builds the normalized state vector S_t plus detailed info metrics.
        """
        q_norm_list = []
        d_norm_list = []

        total_queue_vehicles = 0
        total_queue_length_m = 0.0
        max_queue_length_m = 0.0
        total_waiting_time_s = 0.0
        total_incoming_vehicles = 0

        for lane in self.lanes:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_length = self.lane_lengths[lane]
            total_incoming_vehicles += len(veh_ids)

            queued_dists = []
            for veh_id in veh_ids:
                speed = traci.vehicle.getSpeed(veh_id)
                total_waiting_time_s += traci.vehicle.getWaitingTime(veh_id)

                if speed < self.halt_speed_threshold:
                    pos = traci.vehicle.getLanePosition(veh_id)
                    dist_to_stop = max(0.0, lane_length - pos)
                    queued_dists.append(dist_to_stop)

            n_queued = len(queued_dists)
            queue_len_m = min(lane_length, self._compute_spatial_queue_length(queued_dists))
            q_norm = min(1.0, max(0.0, queue_len_m / lane_length)) if lane_length > 0 else 0.0
            d_norm = min(1.0, max(0.0, float(traci.lane.getLastStepOccupancy(lane))))

            q_norm_list.append(q_norm)
            d_norm_list.append(d_norm)

            total_queue_vehicles += n_queued
            total_queue_length_m += queue_len_m
            if queue_len_m > max_queue_length_m:
                max_queue_length_m = queue_len_m

        waiting_pedestrians = sum(
            1
            for pid in traci.person.getIDList()
            if traci.person.getWaitingTime(pid) > 0.0
        )

        # One-hot encoding of active green phase
        phase_one_hot = [1.0, 0.0] if self.current_green_action == 0 else [0.0, 1.0]
        state = np.array(q_norm_list + d_norm_list + phase_one_hot, dtype=np.float32)

        info = {
            "sim_time": float(traci.simulation.getTime()),
            "phase_action": int(self.current_green_action),
            "sumo_phase": int(traci.trafficlight.getPhase(self.tls_id)),
            "total_queue_vehicles": int(total_queue_vehicles),
            "total_queue_length_m": round(total_queue_length_m, 2),
            "max_queue_length_m": round(max_queue_length_m, 2),
            "sum_q_norm": float(sum(q_norm_list)),
            "total_waiting_time_s": round(total_waiting_time_s, 2),
            "mean_waiting_time_s": round(
                total_waiting_time_s / total_incoming_vehicles, 2
            ) if total_incoming_vehicles > 0 else 0.0,
            "new_stops": int(new_stops),
            "throughput": int(throughput),
            "waiting_pedestrians": int(waiting_pedestrians),
        }

        return state, info

    def _compute_reward(self, step_acc: dict) -> float:
        """
        Time-consistent integral reward R_t invariant to step duration (5 s vs 13 s).
        """
        w = self.reward_weights
        reward = (
            -w["queue"] * step_acc["sum_q_norm_s"]
            - w["wait"] * step_acc["wait_time_s"]
            - w["stops"] * step_acc["new_stops"]
            + w["throughput"] * step_acc["throughput"]
            - w["pedestrian"] * step_acc["wait_peds_s"]
        )
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        binary = "sumo-gui" if self.render_mode == "human" else "sumo"
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

        # Cache lane lengths and simulation time resolution
        self.lane_lengths = {lane: float(traci.lane.getLength(lane)) for lane in self.lanes}
        sim_delta_t = float(traci.simulation.getDeltaT())
        self.steps_per_second = max(1, int(round(1.0 / sim_delta_t)))

        self.step_counter = 0
        self.current_green_action = 0
        self.green_elapsed = 0.0
        self._prev_incoming_vehs = set()
        self._prev_halted_vehs = set()

        # Lock initial NS green phase so SUMO static timer cannot interfere
        self._set_phase_locked(self.GREEN_PHASES[self.current_green_action])

        state, info = self._get_state_and_info()
        return state, info

    def step(self, action):
        target_action = int(action)
        step_acc = {
            "seconds": 0,
            "new_stops": 0,
            "throughput": 0,
            "sum_q_norm_s": 0.0,
            "wait_time_s": 0.0,
            "wait_peds_s": 0,
        }

        def _merge(part: dict) -> None:
            for k in step_acc:
                step_acc[k] += part[k]

        if target_action != self.current_green_action and self.green_elapsed >= self.min_green_time:
            if self.ped_clear_time > 0:
                self._set_phase_locked(self.PED_CLEAR_PHASES[self.current_green_action])
                _merge(self._advance_simulation(self.ped_clear_time))

            if self.yellow_time > 0 and traci.simulation.getTime() < self.max_sim_time:
                self._set_phase_locked(self.YELLOW_PHASES[self.current_green_action])
                _merge(self._advance_simulation(self.yellow_time))

            self.current_green_action = target_action
            self.green_elapsed = 0.0
            self._set_phase_locked(self.GREEN_PHASES[self.current_green_action])

        if traci.simulation.getTime() < self.max_sim_time:
            green_part = self._advance_simulation(self.delta_time)
            self.green_elapsed += green_part["seconds"]
            _merge(green_part)

        self.step_counter += 1

        state, info = self._get_state_and_info(
            new_stops=step_acc["new_stops"],
            throughput=step_acc["throughput"],
        )
        reward = self._compute_reward(step_acc)

        terminated = traci.simulation.getTime() >= self.max_sim_time
        truncated = False

        return state, reward, terminated, truncated, info

    def close(self):
        """
        Safely closes the TraCI connection.
        """
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass