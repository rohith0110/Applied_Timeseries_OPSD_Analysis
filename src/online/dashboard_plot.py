"""
Dashboard generator for the online simulation.
Reads the simulation outputs and creates a comprehensive visualization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

COUNTRY = "DK"
RESULTS_PATH = Path(f"outputs/{COUNTRY}_online_results.csv")
LOGS_PATH = Path(f"outputs/{COUNTRY}_online_updates.csv")
OUTPUT_PLOT = Path(f"outputs/plots/{COUNTRY}_dashboard.png")


def generate_dashboard():
    print(f"Reading results from {RESULTS_PATH}...")
    res_df = pd.read_csv(RESULTS_PATH, parse_dates=["timestamp"])
    log_df = pd.read_csv(LOGS_PATH, parse_dates=["timestamp"])

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    plt.subplots_adjust(hspace=0.1)

    ax = axes[0]
    ax.plot(
        res_df["timestamp"],
        res_df["load"],
        label="Actual Load",
        color="black",
        alpha=0.6,
        linewidth=1,
    )
    ax.plot(
        res_df["timestamp"],
        res_df["yhat"],
        label="Forecast (Median)",
        color="blue",
        alpha=0.8,
        linewidth=1,
    )

    anoms = res_df[res_df["is_anomaly"]]
    ax.scatter(
        anoms["timestamp"], anoms["load"], color="red", label="Anomaly", zorder=5, s=20
    )

    for _, row in log_df.iterrows():
        color = "green" if row["reason"] == "drift" else "gray"
        style = "-" if row["reason"] == "drift" else ":"
        alpha = 0.8 if row["reason"] == "drift" else 0.3
        ax.axvline(row["timestamp"], color=color, linestyle=style, alpha=alpha)

    ax.set_ylabel("Load (MW)")
    ax.set_title(f"Live Simulation Dashboard: {COUNTRY} (Last {len(res_df)} hours)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(
        res_df["timestamp"], res_df["drift_ewma"], label="Drift EWMA", color="purple"
    )

    ax.step(
        log_df["timestamp"],
        log_df["drift_thr"],
        where="post",
        label="Drift Threshold",
        color="orange",
        linestyle="--",
    )

    ax.set_ylabel("Drift Metric")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]

    ax.plot(
        log_df["timestamp"],
        log_df["mase_pre"],
        "o-",
        label="MASE Pre-Adapt",
        color="red",
        markersize=4,
        alpha=0.5,
    )
    ax.plot(
        log_df["timestamp"],
        log_df["mase_post"],
        "o-",
        label="MASE Post-Adapt",
        color="green",
        markersize=4,
        alpha=0.5,
    )

    ax.set_ylabel("MASE (7d)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%H"))
    plt.xticks(rotation=15)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to {OUTPUT_PLOT}")


if __name__ == "__main__":
    generate_dashboard()
