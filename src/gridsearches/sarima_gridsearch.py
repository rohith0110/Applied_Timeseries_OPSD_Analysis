"""
SARIMA Grid Search
------------------
Fast grid search for SARIMA parameters using the existing sarima_cuml.py logic.
Uses subsampled data for speed and writes best parameters back to config.yaml.
"""

import sys
import yaml
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")


os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "1"

import cudf
from cuml.tsa.arima import ARIMA as cuARIMA


sys.path.insert(0, str(Path(__file__).parent.parent))
from classicmodels.sarima_cuml import (
    build_transformed_target,
    build_seasonal_AR_exogs,
    two_stage_fit,
)


def fit_single_sarima_gpu(params, y_raw, country, d, D, s):
    """Fit a single SARIMA model using the existing sarima_cuml two_stage_fit logic."""
    p, q, P, Q = params
    import time

    start = time.time()
    try:

        model, seasonal_info = two_stage_fit(y_raw=y_raw, s=s, P=P, Q=Q, p=p, q=q)

        try:
            aic_val = model.aic
            bic_val = model.bic

            if hasattr(aic_val, "item"):
                aic = float(aic_val.item())
            elif hasattr(aic_val, "iloc"):
                aic = float(aic_val.iloc[0])
            else:
                aic = float(aic_val)

            if hasattr(bic_val, "item"):
                bic = float(bic_val.item())
            elif hasattr(bic_val, "iloc"):
                bic = float(bic_val.iloc[0])
            else:
                bic = float(bic_val)
        except:

            aic = 99999.0
            bic = 99999.0

        elapsed = time.time() - start
        print(
            f"[{country}] ✓ SARIMA({p},{d},{q})({P},{D},{Q})[{s}] in {elapsed:.1f}s → BIC={bic:.1f}"
        )

        return {
            "country": country,
            "p": p,
            "d": d,
            "q": q,
            "P": P,
            "D": D,
            "Q": Q,
            "s": s,
            "AIC": aic,
            "BIC": bic,
        }

    except Exception as e:
        print(f"[{country}] ✗ SARIMA({p},{d},{q})({P},{D},{Q})[{s}] failed: {e}")
        return None


def sarima_gridsearch_from_config(config_path="config.yaml", subsample_months=6):
    """
    Run SARIMA grid search and write best parameters back to config.yaml.

    Args:
        config_path: path to config.yaml
        subsample_months: number of months to use for faster search (default 6)
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(cfg["output_metrics"])
    grid_dir = metrics_dir / "gridsearch"
    grid_dir.mkdir(parents=True, exist_ok=True)

    sarima_cfg = cfg["sarima"]
    d = sarima_cfg["d"]
    D = sarima_cfg["D"]
    s = sarima_cfg["s"]
    p_range = sarima_cfg["p_range"]
    q_range = sarima_cfg["q_range"]
    P_range = sarima_cfg["P_range"]
    Q_range = sarima_cfg["Q_range"]

    print(f"\n{'='*60}")
    print(f"  SARIMA Grid Search (GPU mode)")
    print(f"{'='*60}")
    print(f"Searching: p={p_range}, q={q_range}, P={P_range}, Q={Q_range}")
    print(f"Fixed: d={d}, D={D}, s={s}")
    print(f"Subsample: last {subsample_months} months")
    print(f"{'='*60}\n")

    all_results = []

    for csv_path in cfg["cleaned_files"]:
        country = Path(csv_path).stem
        print(f"\nRunning SARIMA gridsearch for {country}...")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp").asfreq("h")

        if df["load"].isnull().any():
            print(f"   -> {country}: interpolating missing timestamps")
            df["load"] = df["load"].interpolate()

        y = df["load"].values

        subsample_hours = subsample_months * 30 * 24
        if len(y) > subsample_hours:
            print(
                f"   -> Using last {subsample_months} months ({subsample_hours} hours) for {country}"
            )
            y = y[-subsample_hours:]

        param_combos = list(product(p_range, q_range, P_range, Q_range))
        print(f"   -> Testing {len(param_combos)} parameter combinations")

        results = []

        for params in tqdm(param_combos, desc=f"{country}"):
            r = fit_single_sarima_gpu(params, y, country, d, D, s)
            if r:
                results.append(r)

        if not results:
            print(f"No valid SARIMA fits for {country}")
            continue

        df_res = pd.DataFrame(results).sort_values("BIC").reset_index(drop=True)
        df_res.to_csv(grid_dir / f"{country}_sarima_grid.csv", index=False)

        best = df_res.iloc[0].to_dict()
        print(f"\n{'='*60}")
        print(f"Best for {country}:")
        print(
            f"  SARIMA({best['p']},{best['d']},{best['q']})({best['P']},{best['D']},{best['Q']})[{best['s']}]"
        )
        print(f"  BIC = {best['BIC']:.2f}")
        print(f"{'='*60}\n")

        all_results.append(best)

    pd.DataFrame(all_results).to_csv(
        metrics_dir / "sarima_best_models.csv", index=False
    )
    print("\nSARIMA gridsearch completed → sarima_best_models.csv saved")

    if all_results:
        df_all = pd.DataFrame(all_results)

        best_p = int(df_all["p"].mode()[0])
        best_q = int(df_all["q"].mode()[0])
        best_P = int(df_all["P"].mode()[0])
        best_Q = int(df_all["Q"].mode()[0])

        print(f"\n{'='*60}")
        print("Updating config.yaml with best average parameters:")
        print(f"  p={best_p}, d={d}, q={best_q}")
        print(f"  P={best_P}, D={D}, Q={best_Q}, s={s}")
        print(f"{'='*60}\n")

        cfg["sarima_best"] = {
            "p": best_p,
            "d": d,
            "q": best_q,
            "P": best_P,
            "D": D,
            "Q": best_Q,
            "s": s,
        }

        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Config updated: {config_path}")


def main():
    sarima_gridsearch_from_config()


if __name__ == "__main__":
    main()
