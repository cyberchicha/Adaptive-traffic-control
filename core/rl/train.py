import os
import sys
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from core.rl.env import TrafficEnv

def main():
    sumo_cfg_path = os.path.join(ROOT_DIR, "data", "sumo", "cross.sumocfg")
    
    # Creating a folder to save trained models
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    logging.info("Initialization of the environment...")

    log_dir = os.path.join(ROOT_DIR, "data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # render_mode="none" disables the SUMO graphics, so that training can be done faster
    env = TrafficEnv(sumo_cfg_path=sumo_cfg_path, render_mode="none")

    # Monitor automatically records statistics for each episode
    env = Monitor(env, log_dir)

    # Built-in check for SB3,to ensure that the environment is written without errors
    check_env(env)

    logging.info("Creating a DQN Agent...")
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        buffer_size=50000, 
        exploration_fraction=0.1, 
        exploration_final_eps=0.02,
        verbose=1
    )

    new_logger = configure(log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    logging.info("Starting training for 10,000 steps...")
    model.learn(total_timesteps=10000, progress_bar=True)

    model_path = os.path.join(models_dir, "dqn_traffic_agent")
    model.save(model_path)
    logging.info(f"Training completed! The neural network has been saved to {model_path}.zip")

    env.close()

if __name__ == "__main__":
    main()