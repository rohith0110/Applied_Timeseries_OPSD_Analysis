import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def sanity_plot(cleaned_file: Path, output_dir: Path):
    df = pd.read_csv(cleaned_file, parse_dates=["timestamp"])
    last14 = df.iloc[-24 * 14 :]
    start, end = last14["timestamp"].iloc[0], last14["timestamp"].iloc[-1]
    print(f" Starting plot for {cleaned_file.stem} from {start} to {end} ")

    plt.figure(figsize=(16, 9))
    plt.plot(
        last14["timestamp"], last14["load"], label="Load", color="black", linewidth=1
    )
    plt.title(f"Sanity plot for {cleaned_file.stem} ({start} to {end})")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Load")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{cleaned_file.stem}_sanity_plot.png")
    plt.close()


with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

sanity_dir = Path(cfg.get("output_plots")) / "sanity"

csvs = cfg["cleaned_files"]
for csv in csvs:
    sanity_plot(Path(csv), sanity_dir)
    print(f" Sanity plot saved for {csv} ")
print(" All sanity plots done. ")
