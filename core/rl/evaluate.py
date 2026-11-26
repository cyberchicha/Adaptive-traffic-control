import argparse
import csv
import logging
import os
import sys
import sumolib
import traci
from stable_baselines3 import DQN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import BASE_DIR, MODELS_DIR, SUMO_CFG_PATH
from core.rl.env import TrafficEnv

SEED = 42
FIELDNAMES = [
    "step",
    "sim_time",
    "action",
    "sumo_phase",
    "queue_vehicles",
    "queue_length_m",
    "max_queue_length_m",
    "mean_waiting_time_s",
    "new_stops",
    "throughput",
    "waiting_pedestrians",
    "reward",
]


def _summarize_rows(name: str, rows: list) -> dict:
    """
    Aggregates time-weighted and cumulative episode metrics from step logs.
    """
    total_time = 0.0
    prev_time = 0.0

    weighted_wait = 0.0
    weighted_queue_veh = 0.0
    weighted_queue_len = 0.0
    weighted_peds = 0.0

    max_queue_length_m = 0.0
    total_stops = 0
    total_throughput = 0
    total_reward = 0.0

    for r in rows:
        dt = max(1e-6, r["sim_time"] - prev_time)
        prev_time = r["sim_time"]
        total_time += dt

        weighted_wait += r["mean_waiting_time_s"] * dt
        weighted_queue_veh += r["queue_vehicles"] * dt
        weighted_queue_len += r["queue_length_m"] * dt
        weighted_peds += r["waiting_pedestrians"] * dt

        if r["max_queue_length_m"] > max_queue_length_m:
            max_queue_length_m = r["max_queue_length_m"]

        total_stops += r["new_stops"]
        total_throughput += r["throughput"]
        total_reward += r["reward"]

    return {
        "controller": name,
        "mean_waiting_time_s": round(weighted_wait / total_time, 2) if total_time > 0 else 0.0,
        "max_queue_length_m": round(max_queue_length_m, 2),
        "total_stops": int(total_stops),
        "mean_queue_vehicles": round(weighted_queue_veh / total_time, 2) if total_time > 0 else 0.0,
        "mean_queue_length_m": round(weighted_queue_len / total_time, 2) if total_time > 0 else 0.0,
        "mean_waiting_pedestrians": round(weighted_peds / total_time, 2) if total_time > 0 else 0.0,
        "total_throughput": int(total_throughput),
        "total_reward": round(total_reward, 2),
    }


def _save_rows_csv(path, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def run_static_episode(render_mode: str = "none", seed: int = SEED) -> tuple:
    """
    Runs a full 3600 s simulation using SUMO's native fixed-time tlLogic
    and measures the exact same metrics as TrafficEnv.
    """
    env = TrafficEnv(sumo_cfg_path=SUMO_CFG_PATH, render_mode=render_mode)

    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass

    binary = "sumo-gui" if render_mode == "human" else "sumo"
    sumo_cmd = [
        sumolib.checkBinary(binary),
        "-c", str(SUMO_CFG_PATH),
        "--start",
        "--quit-on-end",
        "--no-step-log",
        "--seed", str(int(seed)),
    ]
    traci.start(sumo_cmd)

    env.lane_lengths = {lane: float(traci.lane.getLength(lane)) for lane in env.lanes}
    sim_delta_t = float(traci.simulation.getDeltaT())
    env.steps_per_second = max(1, int(round(1.0 / sim_delta_t)))
    env._prev_incoming_vehs = set()
    env._prev_halted_vehs = set()

    rows = []
    step = 0

    while traci.simulation.getTime() < env.max_sim_time:
        step_acc = env._advance_simulation(env.delta_time)
        _, info = env._get_state_and_info(
            new_stops=step_acc["new_stops"],
            throughput=step_acc["throughput"],
        )
        
        reward = env._compute_reward(step_acc)
        secs = max(1, step_acc["seconds"])
        step += 1

        rows.append({
            "step": step,
            "sim_time": info["sim_time"],
            "action": -1,  # Native static timer
            "sumo_phase": info["sumo_phase"],
            "queue_vehicles": round(step_acc["queue_veh_s"] / secs, 2),
            "queue_length_m": round(step_acc["queue_len_s"] / secs, 2),
            "max_queue_length_m": round(step_acc["max_queue_len_m"], 2),
            "mean_waiting_time_s": round(step_acc["mean_wait_s_sum"] / secs, 2),
            "new_stops": info["new_stops"],
            "throughput": info["throughput"],
            "waiting_pedestrians": round(step_acc["wait_peds_s"] / secs, 2),
            "reward": round(reward, 4),
        })

    env.close()
    summary = _summarize_rows("Static", rows)
    return rows, summary


def run_dqn_episode(model_path, render_mode: str = "none", seed: int = SEED) -> tuple:
    """
    Runs a full 3600 s simulation controlled by the trained DQN agent.
    """
    env = TrafficEnv(sumo_cfg_path=SUMO_CFG_PATH, render_mode=render_mode)
    model = DQN.load(str(model_path))

    obs, _ = env.reset(seed=seed)
    rows = []
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        step += 1

        rows.append({
           "step": step,
            "sim_time": info["sim_time"],
            "action": int(action),
            "sumo_phase": info["sumo_phase"],
            "queue_vehicles": info["step_avg_queue_vehicles"],
            "queue_length_m": info["step_avg_queue_length_m"],
            "max_queue_length_m": info["step_max_queue_length_m"],
            "mean_waiting_time_s": info["step_mean_waiting_time_s"],
            "new_stops": info["new_stops"],
            "throughput": info["throughput"],
            "waiting_pedestrians": info["step_avg_waiting_pedestrians"],
            "reward": round(float(reward), 4),
        })

        if terminated or truncated:
            break

    env.close()
    summary = _summarize_rows("DQN", rows)
    return rows, summary


def add_composite_index(static_summary: dict, dqn_summary: dict) -> None:
    """
    Computes normalized criteria and composite efficiency index (Section 2.4):
      I = 0.5 * W_norm + 0.3 * Q_max_norm + 0.2 * S_norm (lower is better).
    """
    w_ref = max(static_summary["mean_waiting_time_s"], dqn_summary["mean_waiting_time_s"], 1e-9)
    q_ref = max(static_summary["max_queue_length_m"], dqn_summary["max_queue_length_m"], 1e-9)
    s_ref = max(static_summary["total_stops"], dqn_summary["total_stops"], 1e-9)

    for s in (static_summary, dqn_summary):
        w_norm = s["mean_waiting_time_s"] / w_ref
        q_norm = s["max_queue_length_m"] / q_ref
        s_norm = s["total_stops"] / s_ref
        s["W_norm"] = round(w_norm, 4)
        s["Q_max_norm"] = round(q_norm, 4)
        s["S_norm"] = round(s_norm, 4)
        s["composite_index"] = round(0.5 * w_norm + 0.3 * q_norm + 0.2 * s_norm, 4)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Static Traffic Light vs DQN Agent")
    parser.add_argument("--render", action="store_true", help="Run simulations with SUMO GUI")
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "dqn_traffic_agent.zip"),
        help="Path to trained DQN model (.zip)",
    )
    args = parser.parse_args()

    render_mode = "human" if args.render else "none"
    model_path = args.model

    if not os.path.exists(model_path):
        logging.error(f"Model not found: {model_path}")
        return

    data_dir = BASE_DIR / "data"
    static_csv = data_dir / "static_evaluation_metrics.csv"
    dqn_csv = data_dir / "ai_evaluation_metrics.csv"
    summary_csv = data_dir / "benchmark_summary.csv"

    logging.info("1/2 Running baseline Static Traffic Light simulation (3600 s)...")
    static_rows, static_summary = run_static_episode(render_mode=render_mode, seed=SEED)
    _save_rows_csv(static_csv, static_rows)
    logging.info(f"Static metrics saved to {static_csv}")

    logging.info(f"2/2 Running trained DQN agent simulation (3600 s) using {model_path}...")
    dqn_rows, dqn_summary = run_dqn_episode(model_path=model_path, render_mode=render_mode, seed=SEED)
    _save_rows_csv(dqn_csv, dqn_rows)
    logging.info(f"DQN metrics saved to {dqn_csv}")

    add_composite_index(static_summary, dqn_summary)

    # Save summary comparison table
    summary_fields = list(static_summary.keys())
    with open(summary_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(static_summary)
        writer.writerow(dqn_summary)

    # Print formatted comparison table
    print("\n================ BENCHMARK RESULTS (3600 s) ================")
    for key, label in [
        ("mean_waiting_time_s", "Середній час очікування (с/авто)"),
        ("max_queue_length_m", "Максимальна довжина черги (м)"),
        ("total_stops", "Загальна кількість зупинок (шт.)"),
        ("mean_queue_vehicles", "Середня черга (авто)"),
        ("mean_waiting_pedestrians", "Середня кількість очікуючих пішоходів"),
        ("total_throughput", "Пропускна здатність (авто/год)"),
        ("total_reward", "Сумарна винагорода епізоду"),
        ("composite_index", "Комплексний індекс (менше — краще)"),
    ]:
        st_val = static_summary[key]
        dq_val = dqn_summary[key]
        diff_pct = ((dq_val - st_val) / abs(st_val) * 100.0) if st_val != 0 else 0.0
        print(f"{label:40s} | Static: {st_val:8.2f} | DQN: {dq_val:8.2f} | Δ: {diff_pct:+6.1f}%")
    print("============================================================")
    logging.info(f"Summary table saved to {summary_csv}")


if __name__ == "__main__":
    main()