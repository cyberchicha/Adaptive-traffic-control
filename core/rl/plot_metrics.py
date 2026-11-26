import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import BASE_DIR, LOGS_DIR


def plot_training_dynamics():
    train_csv = LOGS_DIR / "train_final" / "progress.csv"
    out_path = BASE_DIR / "data" / "dqn_training_dynamics.png"

    if not os.path.exists(train_csv):
        print(f"[Попередження] Дані навчання не знайдено: {train_csv}")
        return

    df = pd.read_csv(train_csv)
    if "rollout/ep_rew_mean" not in df.columns:
        print("[Попередження] Стовпець rollout/ep_rew_mean відсутній у progress.csv")
        return

    df_clean = df.dropna(subset=["rollout/ep_rew_mean"])

    plt.figure(figsize=(10, 5.5))
    plt.plot(
        df_clean["time/total_timesteps"],
        df_clean["rollout/ep_rew_mean"],
        color="#1f77b4",
        linewidth=2.3,
        label="Середня винагорода за епізод",
    )
    plt.xlabel("Кількість кроків навчання", fontsize=12)
    plt.ylabel("Середня винагорода за епізод", fontsize=12)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Графік динаміки навчання збережено: {out_path}")


def plot_static_vs_dqn():
    data_dir = BASE_DIR / "data"
    static_csv = data_dir / "static_evaluation_metrics.csv"
    dqn_csv = data_dir / "ai_evaluation_metrics.csv"
    summary_csv = data_dir / "benchmark_summary.csv"
    out_path = data_dir / "static_vs_dqn_comparison.png"

    if not (os.path.exists(static_csv) and os.path.exists(dqn_csv) and os.path.exists(summary_csv)):
        print("[Попередження] Спочатку запустіть core/rl/evaluate.py для формування CSV-файлів.")
        return

    df_static = pd.read_csv(static_csv)
    df_dqn = pd.read_csv(dqn_csv)
    df_sum = pd.read_csv(summary_csv).set_index("controller")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # Панель 1: Динаміка кількості автомобілів у черзі протягом 3600 с (з ковзним середнім)
    st_smooth = df_static["queue_vehicles"].rolling(window=12, min_periods=1).mean()
    dq_smooth = df_dqn["queue_vehicles"].rolling(window=10, min_periods=1).mean()

    axes[0].plot(
        df_static["sim_time"],
        st_smooth,
        color="#d62728",
        linewidth=2.0,
        alpha=0.85,
        label=f"Статичний світлофор (сер.: {df_sum.loc['Static', 'mean_queue_vehicles']:.1f} авто)",
    )
    axes[0].plot(
        df_dqn["sim_time"],
        dq_smooth,
        color="#2ca02c",
        linewidth=2.0,
        alpha=0.9,
        label=f"ШІ-агент DQN (сер.: {df_sum.loc['DQN', 'mean_queue_vehicles']:.1f} авто)",
    )
    axes[0].set_xlabel("Час симуляції, с", fontsize=11)
    axes[0].set_ylabel("Кількість автомобілів у черзі (ковзне сер.)", fontsize=11)
    axes[0].legend(fontsize=10, loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Панель 2: Порівняння нормалізованих критеріїв і комплексного індексу (п. 2.4)
    metrics = ["W_norm", "Q_max_norm", "S_norm", "composite_index"]
    labels = [
        "Час очікування\n(W_norm)",
        "Макс. черга\n(Q_max,norm)",
        "Зупинки\n(S_norm)",
        "Комплексний\nіндекс (I)",
    ]

    static_vals = [df_sum.loc["Static", m] for m in metrics]
    dqn_vals = [df_sum.loc["DQN", m] for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    bars_st = axes[1].bar(x - width / 2, static_vals, width, label="Статичний світлофор", color="#d62728", alpha=0.85)
    bars_dq = axes[1].bar(x + width / 2, dqn_vals, width, label="ШІ-агент DQN", color="#2ca02c", alpha=0.9)

    for bar in list(bars_st) + list(bars_dq):
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("Нормалізоване значення (менше — краще)", fontsize=11)
    axes[1].set_ylim(0, 1.18)
    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Графік порівняння збережено: {out_path}")


def main():
    plot_training_dynamics()
    plot_static_vs_dqn()


if __name__ == "__main__":
    main()