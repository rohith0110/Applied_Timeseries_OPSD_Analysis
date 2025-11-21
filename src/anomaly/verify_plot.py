import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages


def plot_verification_samples(verified_path: str):
    verified_path = Path(verified_path)
    if not verified_path.exists():
        print(f"File not found: {verified_path}")
        return

    name_parts = verified_path.stem.replace("anomaly_labels_verified_", "")

    source_path = Path("outputs/anomalies") / f"{name_parts}.csv"
    if not source_path.exists():

        print(f"Could not auto-locate source file at {source_path}")
        print("Please ensure the source anomaly file exists.")
        return

    print(f"Loading samples from: {verified_path}")
    print(f"Loading full data from: {source_path}")

    df_samples = pd.read_csv(verified_path, parse_dates=["timestamp"])
    df_full = pd.read_csv(source_path, parse_dates=["timestamp"])
    df_full = df_full.set_index("timestamp").sort_index()

    out_pdf = verified_path.with_name(f"plots_{verified_path.stem}.pdf")
    print(f"Generating PDF report: {out_pdf} ...")

    with PdfPages(out_pdf) as pdf:
        for idx, row in df_samples.iterrows():
            ts = row["timestamp"]
            label = row.get("silver_label", "N/A")

            start_time = ts - pd.Timedelta(hours=48)
            end_time = ts + pd.Timedelta(hours=48)

            window = df_full.loc[start_time:end_time]

            if window.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                window.index, window["y_true"], label="Actual", color="black", alpha=0.7
            )
            ax.plot(
                window.index, window["yhat"], label="Forecast", color="blue", alpha=0.6
            )

            if "lo" in window.columns and "hi" in window.columns:
                ax.fill_between(
                    window.index,
                    window["lo"],
                    window["hi"],
                    color="blue",
                    alpha=0.1,
                    label="80% PI",
                )

            point = df_full.loc[ts] if ts in df_full.index else None
            if point is not None:
                ax.scatter(
                    [ts],
                    [point["y_true"]],
                    color="red",
                    s=100,
                    zorder=5,
                    label="Target Point",
                )

            ax.set_title(
                f"Sample #{idx+1} | {ts}\nCurrent Label: {label} | Z-score: {row.get('z_resid', 0):.2f}"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.figtext(
                0.5,
                0.01,
                "Action: Check if this looks like an anomaly. Update 'silver_label' in CSV if needed.",
                ha="center",
                fontsize=8,
                style="italic",
            )

            pdf.savefig(fig)
            plt.close(fig)

    print("Done! Open the PDF to review samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the anomaly_labels_verified_*.csv file")
    args = parser.parse_args()

    plot_verification_samples(args.file)
