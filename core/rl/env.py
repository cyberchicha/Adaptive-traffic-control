import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci
import sumolib
import time

class TrafficEnv(gym.Env):
    """
    Custom Environment for Adaptive Traffic Control using SUMO.
    Follows the gymnasium API standard.
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, sumo_cfg_path, render_mode="none"):
        super().__init__()
        self.sumo_cfg_path = sumo_cfg_path
        self.render_mode = render_mode
        
        # 12 approach lanes to the intersection (3 from each side)
        self.lanes = [
            'n2c_0', 'n2c_1', 'n2c_2', 
            's2c_0', 's2c_1', 's2c_2', 
            'e2c_0', 'e2c_1', 'e2c_2', 
            'w2c_0', 'w2c_1', 'w2c_2'
        ]
        self.num_lanes = len(self.lanes)
        self.step_counter = 0
        self.current_total_queue = 0
        
        # ACTION SPACE: 0 - Keep current phase, 1 - Switch to next phase
        self.action_space = spaces.Discrete(2)

        # OBSERVATION SPACE: 3 metrics (N, L, D) for 12 lanes + 1 traffic light phase = 37 values
        obs_shape = (self.num_lanes * 3) + 1
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

    def _get_state(self):
        """
        Reads N, L, D from the simulator (imitating CV module operation).
        """
        state_features = []
        total_queue_count = 0

        for lane in self.lanes:
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)
            lane_length = traci.lane.getLength(lane)

            n = 0
            max_dist = 0.0

            for veh_id in veh_ids:
                speed = traci.vehicle.getSpeed(veh_id)
                
                # Manager's criterion: speed < 5 km/h (1.38 m/s) = queue
                if speed < 1.38:
                    n += 1
                    # Distance from the stop line (end of the lane) to the vehicle
                    pos = traci.vehicle.getLanePosition(veh_id)
                    dist_to_stop = lane_length - pos
                    if dist_to_stop > max_dist:
                        max_dist = dist_to_stop

            l = round(max_dist, 2)
            d = round(n / l, 4) if l > 0 else 0.0

            state_features.extend([n, l, d])
            total_queue_count += n

        # Adding the current traffic light phase
        current_phase = traci.trafficlight.getPhase("center")
        state_features.append(current_phase)

        # Maintaining a common queue for reward calculation
        self.current_total_queue = total_queue_count

        return np.array(state_features, dtype=np.float32)

    def _compute_reward(self):
        """
        Calculates the agent's reward. 
        Minimizing the total vehicle queue and pedestrian waiting time.
        """
        # 1. Penalty for cars (1 car in the queue = -1 point)
        vehicle_penalty = self.current_total_queue

        # 2.Fine for pedestrians
        pedestrian_penalty = 0
        
        # Obtaining a list of all people in the simulation at the current moment
        for person_id in traci.person.getIDList():
            # getWaitingTime returns the time, in seconds, that a person stands motionless
            if traci.person.getWaitingTime(person_id) > 0:
                # Giving pedestrians double priority.
                # 1 person waiting = -2 penalty points
                pedestrian_penalty += 2 

        # Total agent remuneration (the closer to 0, the better)
        reward = -(vehicle_penalty + pedestrian_penalty)
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Closing the previous session if it has frozen
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        # Determining the launch mode (with or without graphics)
        binary = 'sumo-gui' if self.render_mode == "human" else 'sumo'
        sumo_cmd = [sumolib.checkBinary(binary), "-c", self.sumo_cfg_path, "--start", "--quit-on-end"]

        traci.start(sumo_cmd)

        # Reset counters and state variables
        self.step_counter = 0
        self.current_total_queue = 0
        self.time_since_switch = 0  # Timer to prevent instant switching
        
        state = self._get_state()
        return state, {}

    def step(self, action):
        # ENGINEERING SAFEGUARD: Ignore switch command if less than 10 seconds passed
        if action == 1 and getattr(self, 'time_since_switch', 0) < 10:
            action = 0

        # Update the switch timer
        if action == 1:
            self.time_since_switch = 0
        else:
            self.time_since_switch = getattr(self, 'time_since_switch', 0) + 1

        # If the agent decided to switch the traffic light (action 1)
        if action == 1:
            current_phase = traci.trafficlight.getPhase("center")
            # Switch to the next phase (there are 6 phases in our cross.net.xml)
            next_phase = (current_phase + 1) % 6
            traci.trafficlight.setPhase("center", next_phase)

        # Performing 10 micro-steps of 0.1s each = 1 second for the agent
        for _ in range(10):
            traci.simulationStep()
            # Adding a delay between microframes only if graphics are enabled
            if self.render_mode == "human":
                time.sleep(0.03) # 30ms latency (approximately 30 FPS)

        self.step_counter += 1

        state = self._get_state()
        reward = self._compute_reward()

        # The episode ends after 3600 seconds (1 hour of simulation)
        terminated = self.step_counter >= 3600
        truncated = False

        return state, reward, terminated, truncated, {}

    def close(self):
        """
        Safe closing of the TraCI connection.
        """
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass