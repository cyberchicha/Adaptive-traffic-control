import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from config import BASE_DIR, LOGS_DIR


def main():
    experiments = {
        "baseline": "Базова конфігурація (lr=0.001, γ=0.99, ε_frac=0.15)",
        "fast_lr": "Висока швидкість навчання (lr=0.01, γ=0.99)",
        "short_sighted": "Короткий горизонт планування (lr=0.001, γ=0.85)",
        "high_explore": "Розширене дослідження (lr=0.001, ε_frac=0.35)",
    }

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    plt.figure(figsize=(11, 6.5))
    summary = []

    for (exp_name, label), color in zip(experiments.items(), colors):
        csv_path = LOGS_DIR / f"tune_{exp_name}" / "progress.csv"

        if not os.path.exists(csv_path):
            print(f"[Помилка] Файл не знайдено: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "time/total_timesteps" in df.columns and "rollout/ep_rew_mean" in df.columns:
            df_clean = df.dropna(subset=["rollout/ep_rew_mean"])
            if df_clean.empty:
                continue

            plt.plot(
                df_clean["time/total_timesteps"],
                df_clean["rollout/ep_rew_mean"],
                label=label,
                color=color,
                linewidth=2.3,
                alpha=0.9,
            )
            final_rew = df_clean["rollout/ep_rew_mean"].iloc[-1]
            best_rew = df_clean["rollout/ep_rew_mean"].max()
            summary.append((exp_name, final_rew, best_rew))
        else:
            print(f"[Попередження] Необхідні стовпці відсутні у {exp_name}")

    plt.xlabel("Кількість кроків навчання", fontsize=12)
    plt.ylabel("Середня винагорода за епізод", fontsize=12)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plot_path = BASE_DIR / "data" / "tuning_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print("\n--- Підсумок підбору гіперпараметрів ---")
    for name, final_rew, best_rew in sorted(summary, key=lambda x: x[1], reverse=True):
        print(f"{name:15s} | Фінальна сер. винагорода: {final_rew:8.2f} | Максимум: {best_rew:8.2f}")
    print(f"Графік збережено: {plot_path}")


if __name__ == "__main__":
    main()