import os
import sys
import logging

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from core.rl.env import TrafficEnv

PHASE_NAMES = {
    0: "North-South: GREEN",
    1: "North-South: CLEARING (Green is ending)",
    2: "North-South: YELLOW",
    3: "West-East: GREEN",
    4: "West-East: CLEARING (Green is ending)",
    5: "West-East: YELLOW"
}

def main():
    sumo_cfg_path = os.path.join(ROOT_DIR, "data", "sumo", "cross.sumocfg")
    
    if not os.path.exists(sumo_cfg_path):
        logging.error(f"SUMO config not found at: {sumo_cfg_path}")
        return

    logging.info("--- Initialization of RL Environment ---")
    env = TrafficEnv(sumo_cfg_path=sumo_cfg_path, render_mode="human")
    obs, info = env.reset()
    
    logging.info("Starting 60 steps (traffic light switching every 10 seconds)...")
    
    for step in range(60):
        # Keeping the light on for 9 seconds (action 0); at the 10th second, switch it (action 1)
        if step > 0 and step % 10 == 0:
            action = 1
            logging.info(">>> The agent sent a command to switch the traffic light <<<")
        else:
            action = 0
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_phase = int(obs[-1])
        phase_name = PHASE_NAMES.get(current_phase, "Unknown")
        
        # Displaying the status every second
        logging.info(f"Second {step+1:02d} | {phase_name} | Total Queue (Penalty): {reward}")
        
        if terminated or truncated:
            break
            
    env.close()

if __name__ == "__main__":
    main()