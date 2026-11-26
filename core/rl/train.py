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

from config import SUMO_CFG_PATH, MODELS_DIR, LOGS_DIR
from core.rl.env import TrafficEnv

SEED = 42
TOTAL_TIMESTEPS = 100000  # ~180-210 full 1-hour episodes with delta_time=5s

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    log_dir = LOGS_DIR / "train_final"
    os.makedirs(log_dir, exist_ok=True)

    logging.info("Initializing TrafficEnv...")
    env = TrafficEnv(sumo_cfg_path=SUMO_CFG_PATH, render_mode="none")
    env = Monitor(env, str(log_dir))

    # Validate Gymnasium API compliance before training
    check_env(env, warn=True)

    logging.info("Creating DQN agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-2,
        gamma=0.99,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        target_update_interval=2000,
        exploration_fraction=0.15,
        exploration_final_eps=0.02,
        seed=SEED,
        verbose=1,
    )

    new_logger = configure(str(log_dir), ["stdout", "csv"])
    model.set_logger(new_logger)

    logging.info(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

    model_path = MODELS_DIR / "dqn_traffic_agent"
    model.save(str(model_path))
    logging.info(f"Training completed! Model saved to {model_path}.zip")

    env.close()

if __name__ == "__main__":
    main()