import os
import sys
import logging
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import SUMO_CFG_PATH, MODELS_DIR, LOGS_DIR
from core.rl.env import TrafficEnv

SEED = 42
TIMESTEPS = 40000  # ~75-85 full 1-hour episodes with delta_time=5s

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    experiments = [
        {"name": "baseline", "lr": 1e-3, "gamma": 0.99, "explore_frac": 0.15},
        {"name": "fast_lr", "lr": 1e-2, "gamma": 0.99, "explore_frac": 0.15},
        {"name": "short_sighted", "lr": 1e-3, "gamma": 0.85, "explore_frac": 0.15},
        {"name": "high_explore", "lr": 1e-3, "gamma": 0.99, "explore_frac": 0.35},
    ]

    for exp in experiments:
        exp_name = exp["name"]
        logging.info(f"========== STARTING EXPERIMENT: {exp_name.upper()} ==========")

        log_dir = LOGS_DIR / f"tune_{exp_name}"
        os.makedirs(log_dir, exist_ok=True)

        env = TrafficEnv(sumo_cfg_path=SUMO_CFG_PATH, render_mode="none")
        env = Monitor(env, str(log_dir))

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=exp["lr"],
            gamma=exp["gamma"],
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            target_update_interval=2000,
            exploration_fraction=exp["explore_frac"],
            exploration_final_eps=0.02,
            seed=SEED,
            verbose=0,
        )

        new_logger = configure(str(log_dir), ["stdout", "csv"])
        model.set_logger(new_logger)

        logging.info(f"Training {exp_name} for {TIMESTEPS} steps...")
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

        model_path = MODELS_DIR / f"dqn_{exp_name}"
        model.save(str(model_path))
        logging.info(f"Saved {exp_name} model to {model_path}.zip\n")

        env.close()

if __name__ == "__main__":
    main()