"""
Generate example anomaly plots for the report
Creates 2 plots per country:
1. Full time series with all anomalies highlighted
2. Zoomed view of the biggest anomaly event
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("outputs/plots/anomalies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["DK", "ES", "FR"]


def plot_anomaly_overview(df, country, output_path):
    """
    Plot 1: Full time series with anomalies highlighted
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Top subplot: Actual vs Predicted with prediction intervals
    ax1.plot(
        df["timestamp"],
        df["y_true"],
        "b-",
        alpha=0.6,
        linewidth=0.8,
        label="Actual Load",
    )
    ax1.plot(
        df["timestamp"],
        df["yhat"],
        "g-",
        alpha=0.6,
        linewidth=0.8,
        label="Predicted Load",
    )
    ax1.fill_between(
        df["timestamp"],
        df["lo"],
        df["hi"],
        alpha=0.2,
        color="green",
        label="80% Prediction Interval",
    )

    # Highlight flagged anomalies
    anomalies = df[df["flag_z"] == 1]
    if len(anomalies) > 0:
        ax1.scatter(
            anomalies["timestamp"],
            anomalies["y_true"],
            color="red",
            s=50,
            zorder=5,
            label=f"Anomalies (n={len(anomalies)})",
            marker="o",
        )

    ax1.set_ylabel("Load (MW)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"{country} - Energy Load: Actual vs Predicted with Anomaly Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Z-scores with thresholds
    ax2.plot(
        df["timestamp"], df["z_resid"], "k-", alpha=0.5, linewidth=0.8, label="Z-score"
    )
    ax2.axhline(
        y=3, color="orange", linestyle="--", linewidth=1.5, label="±3σ threshold"
    )
    ax2.axhline(y=-3, color="orange", linestyle="--", linewidth=1.5)
    ax2.axhline(
        y=5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="±5σ threshold",
    )
    ax2.axhline(y=-5, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    # Highlight anomaly z-scores
    if len(anomalies) > 0:
        ax2.scatter(
            anomalies["timestamp"],
            anomalies["z_resid"],
            color="red",
            s=50,
            zorder=5,
            marker="o",
        )

    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Z-Score", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"{country} - Residual Z-Scores (Anomaly Threshold Violations)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved overview plot: {output_path}")


def plot_anomaly_detail(df, country, output_path):
    """
    Plot 2: Zoomed view of biggest anomaly event (±3 days window)
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["abs_z"] = df["z_resid"].abs()

    # Find biggest anomaly
    biggest_idx = df["abs_z"].idxmax()
    biggest_row = df.loc[biggest_idx]
    biggest_time = biggest_row["timestamp"]

    # Create ±3 day window
    window_start = biggest_time - pd.Timedelta(days=3)
    window_end = biggest_time + pd.Timedelta(days=3)

    window_df = df[
        (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
    ].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top subplot: Load with prediction intervals
    ax1.plot(
        window_df["timestamp"],
        window_df["y_true"],
        "b-",
        linewidth=2,
        label="Actual Load",
        marker="o",
        markersize=4,
    )
    ax1.plot(
        window_df["timestamp"],
        window_df["yhat"],
        "g--",
        linewidth=2,
        label="Predicted Load",
        marker="s",
        markersize=4,
    )
    ax1.fill_between(
        window_df["timestamp"],
        window_df["lo"],
        window_df["hi"],
        alpha=0.3,
        color="green",
        label="80% Prediction Interval",
    )

    # Highlight THE anomaly
    ax1.scatter(
        [biggest_time],
        [biggest_row["y_true"]],
        color="red",
        s=300,
        zorder=10,
        marker="*",
        edgecolor="darkred",
        linewidth=2,
        label=f'Biggest Anomaly (z={biggest_row["z_resid"]:.2f})',
    )

    # Annotate the anomaly
    ax1.annotate(
        f'Anomaly: {biggest_time.strftime("%Y-%m-%d %H:%M")}\n'
        + f'Actual: {biggest_row["y_true"]:.1f} MW\n'
        + f'Predicted: {biggest_row["yhat"]:.1f} MW\n'
        + f'Deviation: {biggest_row["resid"]:.1f} MW\n'
        + f'Z-score: {biggest_row["z_resid"]:.2f}',
        xy=(biggest_time, biggest_row["y_true"]),
        xytext=(20, 40),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="yellow", alpha=0.8),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=0.3", lw=2, color="red"
        ),
        fontsize=10,
        fontweight="bold",
    )

    ax1.set_ylabel("Load (MW)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"{country} - Detailed View: Largest Anomaly Event (±3 days)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: Residuals and z-scores
    ax2_twin = ax2.twinx()

    # Residuals
    ax2.bar(
        window_df["timestamp"],
        window_df["resid"],
        width=0.03,
        alpha=0.6,
        color="steelblue",
        label="Residual (Actual - Predicted)",
    )
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Z-scores on secondary axis
    ax2_twin.plot(
        window_df["timestamp"],
        window_df["z_resid"],
        "r-",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Z-score",
    )
    ax2_twin.axhline(y=3, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2_twin.axhline(y=-3, color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2_twin.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    # Highlight anomaly point on both axes
    ax2.scatter(
        [biggest_time],
        [biggest_row["resid"]],
        color="red",
        s=200,
        zorder=10,
        marker="*",
        edgecolor="darkred",
        linewidth=2,
    )
    ax2_twin.scatter(
        [biggest_time],
        [biggest_row["z_resid"]],
        color="darkred",
        s=200,
        zorder=10,
        marker="*",
        edgecolor="black",
        linewidth=2,
    )

    ax2.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residual (MW)", fontsize=12, fontweight="bold", color="steelblue")
    ax2_twin.set_ylabel("Z-Score", fontsize=12, fontweight="bold", color="red")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2_twin.tick_params(axis="y", labelcolor="red")

    # Legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved detail plot: {output_path}")
    print(
        f"  → Anomaly: {biggest_time.strftime('%Y-%m-%d %H:%M')}, z={biggest_row['z_resid']:.2f}"
    )


def main():
    print("=" * 70)
    print("GENERATING ANOMALY EXAMPLE PLOTS FOR REPORT")
    print("=" * 70)

    for country in COUNTRIES:
        print(f"\n[{country}] Processing...")

        # Load ensemble anomaly data
        anomaly_file = f"outputs/anomalies/{country}_anomalies_ensemble.csv"

        try:
            df = pd.read_csv(anomaly_file)
            print(f"  Loaded {len(df)} data points from {anomaly_file}")
            print(f"  Anomalies flagged: {df['flag_z'].sum()}")

            # Generate plots
            overview_path = OUTPUT_DIR / f"{country}_anomaly_overview.png"
            detail_path = OUTPUT_DIR / f"{country}_anomaly_detail_biggest.png"

            plot_anomaly_overview(df, country, overview_path)
            plot_anomaly_detail(df, country, detail_path)

        except FileNotFoundError:
            print(f"  ⚠ Warning: {anomaly_file} not found, skipping {country}")
        except Exception as e:
            print(f"  ✗ Error processing {country}: {e}")

    print("\n" + "=" * 70)
    print("DONE! Plots saved to:", OUTPUT_DIR.absolute())
    print("=" * 70)
    print("\nGenerated plots:")
    print("  1. *_anomaly_overview.png     → Full time series with all anomalies")
    print("  2. *_anomaly_detail_biggest.png → Zoomed view of largest anomaly")


if __name__ == "__main__":
    main()
