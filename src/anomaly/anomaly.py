import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml


def rolling_zscore(
    residuals: pd.Series, window: int = 336, min_periods: int = 168
) -> pd.Series:
    """Compute rolling z-score for a pandas Series of residuals.

    Returns a Series aligned with the input (NaN where insufficient data).
    """
    if not isinstance(residuals, pd.Series):
        residuals = pd.Series(residuals)

    rol_mean = residuals.rolling(window=window, min_periods=min_periods).mean()
    rol_std = residuals.rolling(window=window, min_periods=min_periods).std()
    z = (residuals - rol_mean) / rol_std
    return z


def cusum_test(z_series: pd.Series, k: float = 0.5, h: float = 5.0) -> np.ndarray:
    """A simple one-sided CUSUM implementation that is NaN-robust.

    Returns an int array (0/1) of the same length as `z_series` where 1 indicates
    an alarm at that timepoint.
    """
    z = np.asarray(z_series)
    s_pos = 0.0
    s_neg = 0.0
    alarms = np.zeros(len(z), dtype=int)

    for i, val in enumerate(z):
        if np.isnan(val):

            s_pos = s_pos
            s_neg = s_neg
            alarms[i] = 0
            continue

        s_pos = max(0.0, s_pos + val - k)
        s_neg = min(0.0, s_neg + val + k)

        if s_pos > h or abs(s_neg) > h:
            alarms[i] = 1

            s_pos = 0.0
            s_neg = 0.0
        else:
            alarms[i] = 0

    return alarms


def process_country_df(df: pd.DataFrame, cc: str, output_path: Path) -> pd.DataFrame:
    """Process a concatenated dataframe of test forecasts for a single country.

    Writes three CSVs to `output_path` (per-model, ensemble, and sarima-only when applicable)
    and returns the ensemble DataFrame.
    """

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    required = ["y_true", "yhat"]
    if not all(c in df.columns for c in required):
        print(f"⚠️  Missing required columns for {cc}: {required}")
        return pd.DataFrame()

    df = df[~(df["y_true"].isna() & df["yhat"].isna())].copy()
    if df.empty:
        print(f"⚠️  No valid forecast rows for {cc}")
        return pd.DataFrame()

    df["resid"] = df["y_true"] - df["yhat"]

    per_model_rows = []
    model_types = df["model_type"].unique() if "model_type" in df.columns else ["model"]
    for m in model_types:
        mask = df.get("model_type", pd.Series([m] * len(df))) == m
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        sub["z_resid"] = rolling_zscore(sub["resid"])
        sub["flag_z"] = ((sub["z_resid"].abs() >= 3) & sub["z_resid"].notna()).astype(
            int
        )
        flags = cusum_test(sub["z_resid"]) if len(sub) > 0 else np.array([])
        if len(flags) == len(sub):
            flags = np.where(sub["z_resid"].isna(), 0, flags)
            sub["flag_cusum"] = flags
        else:
            sub["flag_cusum"] = 0
        sub["model_type"] = m
        per_model_rows.append(sub)

    if len(per_model_rows) > 0:
        per_model_df = pd.concat(per_model_rows, ignore_index=True)
        out_per_model = output_path / f"{cc}_anomalies_per_model.csv"
        cols_out = [
            "timestamp",
            "model_type",
            "y_true",
            "yhat",
            "lo" if "lo" in per_model_df.columns else None,
            "hi" if "hi" in per_model_df.columns else None,
            "horizon" if "horizon" in per_model_df.columns else None,
            "train_end" if "train_end" in per_model_df.columns else None,
            "resid",
            "z_resid",
            "flag_z",
            "flag_cusum",
        ]
        cols_out = [c for c in cols_out if c is not None]
        per_model_df.to_csv(out_per_model, index=False, columns=cols_out)
        print(
            f"✓ Saved per-model anomalies → {out_per_model} (rows={len(per_model_df)})"
        )
    else:
        per_model_df = pd.DataFrame()

    grouped = df.groupby("timestamp")
    try:
        y_true = grouped["y_true"].first()
    except Exception:
        y_true = grouped["y_true"].apply(
            lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan
        )

    yhat_med = grouped["yhat"].median()
    lo80 = grouped["yhat"].quantile(0.10)
    hi80 = grouped["yhat"].quantile(0.90)

    ens = pd.DataFrame(
        {
            "timestamp": yhat_med.index,
            "y_true": y_true.values,
            "yhat": yhat_med.values,
            "lo": lo80.values,
            "hi": hi80.values,
        }
    )
    ens = ens.sort_values("timestamp").reset_index(drop=True)
    ens["resid"] = ens["y_true"] - ens["yhat"]
    ens["z_resid"] = rolling_zscore(ens["resid"])
    ens["flag_z"] = ((ens["z_resid"].abs() >= 3) & ens["z_resid"].notna()).astype(int)
    flags = cusum_test(ens["z_resid"]) if len(ens) > 0 else np.array([])
    if len(flags) == len(ens):
        flags = np.where(ens["z_resid"].isna(), 0, flags)
        ens["flag_cusum"] = flags
    else:
        ens["flag_cusum"] = 0

    out_ens = output_path / f"{cc}_anomalies_ensemble.csv"
    cols_out = [
        "timestamp",
        "y_true",
        "yhat",
        "lo",
        "hi",
        "resid",
        "z_resid",
        "flag_z",
        "flag_cusum",
    ]
    ens.to_csv(out_ens, index=False, columns=cols_out)

    out_legacy = output_path / f"{cc}_anomalies.csv"
    ens.to_csv(out_legacy, index=False, columns=cols_out)
    print(f"✓ Saved ensemble anomalies → {out_ens} (rows={len(ens)})")
    print(f"✓ Saved legacy anomalies → {out_legacy} (ensemble copy)")

    sarima_candidates = [
        m
        for m in model_types
        if ("sarima" in str(m).lower()) or ("sarimax" in str(m).lower())
    ]
    if len(sarima_candidates) > 0:
        sarima_model = sarima_candidates[0]
        sar_mask = (
            df.get("model_type", pd.Series([sarima_model] * len(df))) == sarima_model
        )
        sar = df[sar_mask].copy()
        if len(sar) > 0:
            sar = sar.sort_values("timestamp").reset_index(drop=True)
            sar["z_resid"] = rolling_zscore(sar["resid"])
            sar["flag_z"] = (
                (sar["z_resid"].abs() >= 3) & sar["z_resid"].notna()
            ).astype(int)
            flags = cusum_test(sar["z_resid"]) if len(sar) > 0 else np.array([])
            if len(flags) == len(sar):
                flags = np.where(sar["z_resid"].isna(), 0, flags)
                sar["flag_cusum"] = flags
            else:
                sar["flag_cusum"] = 0

            out_sar = output_path / f"{cc}_anomalies_sarima.csv"
            cols_out = [
                "timestamp",
                "model_type",
                "y_true",
                "yhat",
                "lo" if "lo" in sar.columns else None,
                "hi" if "hi" in sar.columns else None,
                "horizon" if "horizon" in sar.columns else None,
                "train_end" if "train_end" in sar.columns else None,
                "resid",
                "z_resid",
                "flag_z",
                "flag_cusum",
            ]
            cols_out = [c for c in cols_out if c is not None]
            sar.to_csv(out_sar, index=False, columns=cols_out)
            print(
                f"✓ Saved SARIMA-only anomalies ({sarima_model}) → {out_sar} (rows={len(sar)})"
            )
    else:
        print(
            f"⚠️  No SARIMA candidate model found for {cc}; skipping SARIMA-only anomalies."
        )

    return ens


def main(config_path: str = None):

    project_root = Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path) if config_path else project_root / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    forecasts_dir = Path(cfg.get("output_dir", "outputs")) / "forecasts"
    out_dir = Path(cfg.get("output_anomalies", "outputs/anomalies"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(forecasts_dir.glob("*_test.csv"))
    if len(all_files) == 0:
        print(f"No forecast files found in {forecasts_dir} matching '*_test.csv'")
        return

    files_by_cc = {}
    for p in all_files:
        cc = p.stem.split("_")[0]
        files_by_cc.setdefault(cc, []).append(p)

    for cc, paths in tqdm(files_by_cc.items(), desc="Processing countries"):
        print(f"\n=== Aggregating {len(paths)} files for country: {cc} ===")
        dfs = []
        for p in paths:
            print(f"  - reading {p.name}")
            try:
                d = pd.read_csv(p, parse_dates=["timestamp"])
                if "split" in d.columns:
                    d = d[d["split"] == "test"].copy()
                dfs.append(d)
            except Exception as e:
                print(f"  ⚠️  Failed to read {p.name}: {e}")

        if len(dfs) == 0:
            print(f"⚠️  No test rows found for {cc}; skipping anomaly file.")
            continue

        df_all = pd.concat(dfs, ignore_index=True)
        process_country_df(df_all, cc, out_dir)


if __name__ == "__main__":
    main()
