import os
import sys
import logging
import numpy as np

# Logging settings
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from core.rl.env import TrafficEnv
from config import SUMO_CFG_PATH

PHASE_NAMES = {
    0: "North-South: GREEN",
    1: "North-South: PED CLEAR",
    2: "North-South: YELLOW",
    3: "East-West: GREEN",
    4: "East-West: PED CLEAR",
    5: "East-West: YELLOW",
}

def main():
    if not os.path.exists(SUMO_CFG_PATH):
        logging.error(f"SUMO config not found at: {SUMO_CFG_PATH}.")
        return

    logging.info("--- Initialization of RL Environment ---")
    env = TrafficEnv(sumo_cfg_path=SUMO_CFG_PATH, render_mode="human")
    obs, info = env.reset()
    
    assert obs.shape == (18,), f"Expected obs shape (18,), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "State vector out of [0, 1] bounds"
    logging.info(f"Initial obs verified: shape={obs.shape}, min={obs.min():.3f}, max={obs.max():.3f}.")

    # Direct phase schedule to test both normal switching and min_green_time=10s safeguard:
    # Step 1: request EW (1) at 0s -> must be IGNORED (stays NS)
    # Step 2: keep NS (0) -> green_elapsed reaches 10s
    # Step 3: request EW (1) -> triggers 5s ped_clear + 3s yellow + 5s EW green
    # Then toggle every 3 steps (15s green per direction)
    target_action = 0

    logging.info("Starting 15 RL steps (delta_time=5s)...")

    for step in range(1, 16):
        if step == 1:
            action = 1
            logging.info(">>> Testing min_green_time safeguard: requesting early switch to EW <<<")
        elif step == 2:
            action = 0
        elif step % 3 == 0:
            target_action = 1 - target_action
            action = target_action
            direction = "EW" if action == 1 else "NS"
            logging.info(f">>> Agent requested switch to {direction} (action={action}) <<<")
        else:
            action = target_action

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (18,), f"Unexpected obs shape: {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "State vector out of [0, 1] bounds"

        phase_name = PHASE_NAMES.get(info["sumo_phase"], "Unknown")
        logging.info(
            f"Step {step:02d} | t={info['sim_time']:5.1f}s | {phase_name:24s} | "
            f"Q={info['total_queue_vehicles']:2d} veh ({info['total_queue_length_m']:5.1f}m) | "
            f"Thr={info['throughput']:2d} | Stops={info['new_stops']:2d} | "
            f"Peds={info['waiting_pedestrians']:2d} | R={reward:6.2f}"
        )

        if terminated or truncated:
            break

    env.close()
    logging.info("Environment test completed successfully.")

if __name__ == "__main__":
    main()