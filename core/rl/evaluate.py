import os
import sys
import logging
import time
import csv
from stable_baselines3 import DQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from core.rl.env import TrafficEnv

def main():
    sumo_cfg_path = os.path.join(ROOT_DIR, "data", "sumo", "cross.sumocfg")
    model_path = os.path.join(ROOT_DIR, "models", "dqn_traffic_agent.zip")
    
    metrics_path = os.path.join(ROOT_DIR, "data", "ai_evaluation_metrics.csv")

    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return

    logging.info("Initializing environment...")
    env = TrafficEnv(sumo_cfg_path=sumo_cfg_path, render_mode="human")
    model = DQN.load(model_path)

    obs, info = env.reset()
    
    logging.info(f"Starting simulation. Metrics will be saved to {metrics_path}")

    # Opening CSV file for writing
    with open(metrics_path, mode='w', newline='') as csv_file:
        fieldnames = ['step', 'action', 'phase', 'penalty']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for step in range(200):  # Starting the simulation for 200 steps
            # The agent makes a conscious decision
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_phase = int(obs[-1])
            
            # Writing a row with metrics
            writer.writerow({
                'step': step + 1,
                'action': action,
                'phase': current_phase,
                'penalty': float(reward)
            })

            logging.info(f"Step {step+1:03d} | Action: {action} | Phase: {current_phase} | Penalty: {reward}")
            
            # Small delay for visual comfort
            time.sleep(0.2)

            if terminated or truncated:
                break

    logging.info("Testing completed. Data saved!")
    env.close()

if __name__ == "__main__":
    main()