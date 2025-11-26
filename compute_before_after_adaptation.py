"""
Compute Before/After Adaptation Metrics for Live Simulation
Compares static baseline model vs adaptive model with periodic retraining
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_mase(y_true, y_pred, y_train):
    """Compute Mean Absolute Scaled Error"""
    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    return mae / (naive_mae + 1e-10)


def compute_coverage(y_true, y_lo, y_hi):
    """Compute prediction interval coverage"""
    in_bounds = (y_true >= y_lo) & (y_true <= y_hi)
    return np.mean(in_bounds) * 100


def rolling_window_metrics(df, window_hours=168):
    """Compute rolling window MASE and coverage"""
    results = []

    for i in range(window_hours, len(df)):
        window = df.iloc[i - window_hours : i]

        # MASE using 24-hour naive baseline
        y_true = window["load"].values
        y_pred = window["yhat"].values

        if len(y_true) > 24:
            naive_baseline = y_true[:-24]
            actual_for_mase = y_true[24:]
            mase = compute_mase(actual_for_mase, y_pred[24:], naive_baseline)
        else:
            mase = np.nan

        # Coverage
        coverage = compute_coverage(
            y_true, window["yhat_lo"].values, window["yhat_hi"].values
        )

        results.append(
            {
                "timestamp": window.iloc[-1]["timestamp"],
                "mase": mase,
                "coverage": coverage,
            }
        )

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("COMPUTING BEFORE/AFTER ADAPTATION METRICS")
    print("=" * 80)

    COUNTRIES = ["DK", "ES", "FR"]

    for country in COUNTRIES:
        print(f"\n[{country}] Processing...")

        # Load online simulation results (WITH adaptation)
        online_path = f"outputs/{country}_online_results.csv"

        try:
            df_adaptive = pd.read_csv(online_path, parse_dates=["timestamp"])
            print(f"  Loaded {len(df_adaptive)} hours of adaptive simulation data")

            # For baseline comparison, we need to simulate what would happen WITHOUT updates
            # The first ~168 hours are "pure" baseline before first update
            # We'll use those to estimate baseline performance

            # Compute rolling metrics for adaptive model
            print("  Computing rolling 7-day metrics for ADAPTIVE model...")
            adaptive_metrics = rolling_window_metrics(df_adaptive, window_hours=168)

            # Extract baseline period (first 7 days before major updates)
            baseline_period = df_adaptive.iloc[: 168 * 2].copy()  # First 2 weeks
            baseline_metrics = rolling_window_metrics(baseline_period, window_hours=168)

            # Compute overall averages
            baseline_mase = baseline_metrics["mase"].mean()
            baseline_coverage = baseline_metrics["coverage"].mean()

            # Adaptive period (after updates have kicked in - last 70% of data)
            adaptation_start_idx = int(len(adaptive_metrics) * 0.3)
            adapted_metrics = adaptive_metrics.iloc[adaptation_start_idx:]

            adaptive_mase = adapted_metrics["mase"].mean()
            adaptive_coverage = adapted_metrics["coverage"].mean()

            # Compute improvement
            mase_improvement = ((baseline_mase - adaptive_mase) / baseline_mase) * 100
            coverage_improvement = adaptive_coverage - baseline_coverage

            print(f"\n  RESULTS for {country}:")
            print(
                f"  {'Metric':<25} {'Baseline (Static)':<20} {'Adaptive (Online)':<20} {'Improvement':<15}"
            )
            print(f"  {'-'*80}")
            print(
                f"  {'Rolling 7-day MASE':<25} {baseline_mase:<20.4f} {adaptive_mase:<20.4f} {mase_improvement:>+6.1f}%"
            )
            print(
                f"  {'80% PI Coverage (%)':<25} {baseline_coverage:<20.2f} {adaptive_coverage:<20.2f} {coverage_improvement:>+6.1f}pp"
            )

            # Save detailed results
            output_path = f"outputs/metrics/{country}_adaptation_comparison.csv"
            Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

            comparison_df = pd.DataFrame(
                [
                    {
                        "country": country,
                        "metric": "MASE",
                        "baseline_static": baseline_mase,
                        "adaptive_online": adaptive_mase,
                        "improvement_pct": mase_improvement,
                    },
                    {
                        "country": country,
                        "metric": "Coverage",
                        "baseline_static": baseline_coverage,
                        "adaptive_online": adaptive_coverage,
                        "improvement_pp": coverage_improvement,
                    },
                ]
            )

            comparison_df.to_csv(output_path, index=False)
            print(f"  ✓ Saved comparison to {output_path}")

            # Also save full rolling metrics for plotting
            adaptive_metrics.to_csv(
                f"outputs/metrics/{country}_rolling_metrics.csv", index=False
            )
            print(
                f"  ✓ Saved rolling metrics to outputs/metrics/{country}_rolling_metrics.csv"
            )

        except FileNotFoundError:
            print(f"  ⚠ {online_path} not found, skipping")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 80)
    print("GENERATING SUMMARY TABLE FOR REPORT")
    print("=" * 80 + "\n")

    # Create summary table for all countries
    summary_rows = []
    for country in COUNTRIES:
        try:
            comp_df = pd.read_csv(
                f"outputs/metrics/{country}_adaptation_comparison.csv"
            )
            mase_row = comp_df[comp_df["metric"] == "MASE"].iloc[0]
            cov_row = comp_df[comp_df["metric"] == "Coverage"].iloc[0]

            summary_rows.append(
                {
                    "Country": country,
                    "Baseline MASE": f"{mase_row['baseline_static']:.3f}",
                    "Adaptive MASE": f"{mase_row['adaptive_online']:.3f}",
                    "MASE Δ": f"{mase_row['improvement_pct']:+.1f}%",
                    "Baseline Coverage": f"{cov_row['baseline_static']:.1f}%",
                    "Adaptive Coverage": f"{cov_row['adaptive_online']:.1f}%",
                    "Coverage Δ": f"{cov_row['improvement_pp']:+.1f}pp",
                }
            )
        except:
            pass

    summary_df = pd.DataFrame(summary_rows)

    print("\nSUMMARY TABLE (Copy to Report):\n")
    print(summary_df.to_markdown(index=False))

    summary_df.to_csv("outputs/metrics/adaptation_summary_table.csv", index=False)
    print(f"\n✓ Summary table saved to outputs/metrics/adaptation_summary_table.csv")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
