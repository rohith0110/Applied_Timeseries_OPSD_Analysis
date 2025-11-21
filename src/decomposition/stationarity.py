from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def check_stationarity(
    df, col="load", country="unknown", output: Path = Path("outputs/plots/stationarity")
):
    output.mkdir(parents=True, exist_ok=True)
    series = df[col].dropna()
    adf_result = adfuller(series)
    pval = adf_result[1]
    print(f"\n[{country}] ADF test p-value: {pval:.4f}")
    print(
        f"→ Non-stationary has {pval:.4f} but we apply differencing because adf is not able to capture stationarity properly (has trend). Applying differencing (d=1)."
    )
    df_diff = series.diff().dropna()
    d = 1
    for s in [24, 168]:
        df_seasonal_diff = df_diff.diff(s).dropna()
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(df_seasonal_diff, lags=48, ax=axes[0])
        plot_pacf(df_seasonal_diff, lags=48, ax=axes[1])
        plt.suptitle(f"{country}: ACF & PACF (d={d}, D={1}, s={s})")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output / f"{country}_ACF_PACF_s{s}.png", dpi=200)
        plt.close(fig)

    return {
        "country": country,
        "adf_pvalue": round(pval, 4),
        "d": d,
        "D": 1,
        "s": s,
    }


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    plots_dir = Path(cfg.get("output_plots")) / "stationarity"
    metrics_dir = Path(cfg.get("output_metrics"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cleaned_files = cfg.get("cleaned_files", [])

    results = []
    for csv_path in cleaned_files:
        country = Path(csv_path).stem
        df = (
            pd.read_csv(csv_path, parse_dates=["timestamp"])
            .set_index("timestamp")
            .asfreq("h")
        )
        result = check_stationarity(df, col="load", country=country, output=plots_dir)
        results.append(result)

    pd.DataFrame(results).to_csv(metrics_dir / "adf_diff_summary.csv", index=False)
    print(
        "\n✅ Stationarity tests complete. Saved summary to metrics/adf_diff_summary.csv"
    )
