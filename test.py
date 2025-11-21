from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def difference_and_check(csv_path: str, output_dir="outputs", s=24):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    country = Path(csv_path).stem
    df = (
        pd.read_csv(csv_path, parse_dates=["timestamp"])
        .set_index("timestamp")
        .asfreq("1H")
    )

    # --- 1️⃣ Plot original series ---
    plt.figure(figsize=(12, 4))
    plt.plot(df["load"], color="steelblue")
    plt.title(f"{country}: Original Series (Non-stationary)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{country}_original_series.png", dpi=200)
    plt.close()

    # --- 2️⃣ First differencing (remove trend) ---
    df_diff = df["load"].diff().dropna()

    plt.figure(figsize=(12, 4))
    plt.plot(df_diff, color="orange")
    plt.title(f"{country}: After 1st Differencing (d=1)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{country}_diff1.png", dpi=200)
    plt.close()

    # --- 3️⃣ Seasonal differencing (remove daily seasonality) ---
    df_seasonal_diff = df_diff.diff(168).dropna()

    plt.figure(figsize=(12, 4))
    plt.plot(df_seasonal_diff, color="green")
    plt.title(f"{country}: After Seasonal Differencing (D=1, s={s})")
    plt.tight_layout()
    plt.savefig(output_dir / f"{country}_diff_seasonal.png", dpi=200)
    plt.close()

    # --- 4️⃣ ADF tests before/after ---
    def adf_test(series, label):
        result = adfuller(series.dropna())
        pval = result[1]
        print(f"{label} ADF p-value: {pval:.5f}")
        return pval

    print(f"\n=== {country} Stationarity Checks ===")
    p_raw = adf_test(df["load"], "Raw series")
    p_diff = adf_test(df_diff, "After 1st differencing")
    p_seasonal = adf_test(df_seasonal_diff, "After 1st + seasonal differencing")

    # Save ADF results
    pd.DataFrame(
        {
            "stage": ["raw", "diff1", "diff1+seasonal"],
            "adf_pvalue": [p_raw, p_diff, p_seasonal],
        }
    ).to_csv(output_dir / f"{country}_adf_differencing.csv", index=False)

    print(f"Plots and ADF summary saved for {country}\n")


# Example run
difference_and_check("data/DK_cleaned.csv", output_dir="outputs", s=24)
