import os
import pandas as pd
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
outpaths = []
countries = cfg["countries"]
data_path = cfg["data_path"]
output_dir = cfg["data"]
os.makedirs(output_dir, exist_ok=True)

timestamp = cfg["columns"]["timestamp"]
load = cfg["columns"]["load_pattern"]
solar = cfg["columns"]["solar_pattern"]
wind = cfg["columns"]["wind_pattern"]

print("loading the main csv dataset...")
df = pd.read_csv(data_path, parse_dates=[timestamp])
print(f"shape: {df.shape[0]} rows, {df.shape[1]} columns")

for cc in countries:
    print(f"\nprocessing {cc}")
    col_map = {}
    col_candidates = {
        "load": load.format(cc=cc),
        "solar": solar.format(cc=cc),
        "wind": wind.format(cc=cc),
    }
    existing_cols = [timestamp] + [
        v for v in col_candidates.values() if v in df.columns
    ]
    print(f"columns for cleaned data: {existing_cols}")
    if len(existing_cols) <= 1:
        print(f"no data columns found for {cc}, skipping.")
        continue

    sub = df[existing_cols].copy()
    rename = {timestamp: "timestamp"}
    for k, v in col_candidates.items():
        if v in sub.columns:
            rename[v] = k
    sub.rename(columns=rename, inplace=True)
    sub = sub.dropna(subset=["load"])
    sub = sub.interpolate(limit_direction="both")
    sub = sub.sort_values("timestamp").set_index("timestamp").asfreq("h")
    sub = sub.interpolate(limit_direction="both").reset_index()

    if sub.empty:
        print(f"no data left for {cc} after processing, skipping.")
        continue
    print(f"{cc}: {len(sub)} rows, {len(sub.columns)} columns")
    out = os.path.join(output_dir, f"{cc}_cleaned.csv")
    sub.to_csv(out, index=False)
    outpaths.append(out)
    print(f"Saved clean {cc} data to {out}")
cfg["cleaned_files"] = outpaths
with open("config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)
print(f"\nDone for countries: {countries}.")
