from pathlib import Path
import pandas as pd
import yaml
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def STLPOT(c_file: Path, output_dir: Path):
    df = pd.read_csv(c_file, parse_dates=["timestamp"])
    df = df[["timestamp", "load"]].set_index("timestamp")
    output_dir = output_dir / c_file.stem.split("_")[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    stl = STL(df["load"], period=24, robust=True, seasonal=25, trend=365, low_pass=25)
    res = stl.fit()

    fig = res.plot()

    plt.suptitle(f"STL Decomposition(Full) for {c_file.stem}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f"{c_file.stem}_STL.png", dpi=200)
    plt.close(fig)

    window = 24 * 60
    idx = df.index[-window:]
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(idx, res.observed[-window:], color="black", label="Observed")
    axes[1].plot(idx, res.trend[-window:], color="orange", label="Trend")
    axes[2].plot(idx, res.seasonal[-window:], color="green", label="Seasonal")
    axes[3].plot(idx, res.resid[-window:], color="red", label="Residual")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"STL Decomposition last 60 days for {c_file.stem}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f"{c_file.stem}_STL_60.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(res.resid.dropna(), lags=200, ax=ax)
    ax.set_title(
        f"ACF of Residuals – {c_file.stem}\n"
        "Strong correlation at lag ≈168 → Weekly Seasonality not captured by 24h STL"
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{c_file.stem}_ACF_Residuals.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    stl_dir = Path(cfg.get("output_plots")) / "stl_acf"

    for csv in cfg["cleaned_files"]:
        STLPOT(Path(csv), stl_dir)
        print(f" STL decomposition plot saved for {csv} ")
    print(" All STL decomposition plots done. ")
