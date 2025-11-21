from functools import partial
import yaml
from pathlib import Path
from itertools import product
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")


def fit_single_arima(params, y, country, d):
    """Fit ARIMA(p,d,q) from config ranges."""
    p, q = params
    import time

    start = time.time()

    try:
        model = SARIMAX(
            y, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False, method="lbfgs", maxiter=100, concentrate_scale=True)
        elapsed = time.time() - start
        print(f"[{country}] ✓ ARIMA({p},{d},{q}) in {elapsed:.1f}s")

        return {
            "country": country,
            "p": p,
            "d": d,
            "q": q,
            "AIC": model.aic,
            "BIC": model.bic,
        }

    except Exception:
        return None


def arima_gridsearch_from_config(
    config_path="config.yaml", use_parallel=True, subsample_data=True
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(cfg["output_metrics"])
    grid_dir = metrics_dir / "gridsearch"
    grid_dir.mkdir(parents=True, exist_ok=True)

    arima_cfg = cfg["arima"]
    d = arima_cfg["d"]
    p_range = arima_cfg["p_range"]
    q_range = arima_cfg["q_range"]

    all_results = []

    for csv_path in cfg["cleaned_files"]:
        country = Path(csv_path).stem
        df = (
            pd.read_csv(csv_path, parse_dates=["timestamp"])
            .set_index("timestamp")
            .asfreq("h")
        )

        if df["load"].isnull().any():
            print(f"   -> {country}: interpolating missing")
            df["load"] = df["load"].interpolate()

        y = df["load"].dropna()

        if subsample_data and len(y) > 17280:
            print(f"   -> Using last 6 months for ARIMA ({country})")
            y = y.iloc[-17280:]

        print(f"\nRunning ARIMA gridsearch for {country}...")

        param_combos = list(product(p_range, q_range))
        results = []

        if use_parallel:
            n_cores = max(1, mp.cpu_count() - 1)
            print(f"   -> Using {n_cores} cores")

            fit_func = partial(fit_single_arima, y=y, country=country, d=d)

            with mp.Pool(n_cores) as pool:
                raw = list(
                    tqdm(
                        pool.imap(fit_func, param_combos),
                        total=len(param_combos),
                        desc=f"{country}",
                    )
                )
            results = [r for r in raw if r is not None]

        else:
            for params in tqdm(param_combos, desc=f"{country}"):
                r = fit_single_arima(params, y, country, d)
                if r:
                    results.append(r)

        if not results:
            print(f"No ARIMA fits for {country}")
            continue

        df_res = pd.DataFrame(results).sort_values("BIC").reset_index(drop=True)
        df_res.to_csv(grid_dir / f"{country}_arima_grid.csv", index=False)

        best = df_res.iloc[0].to_dict()
        print(
            f"Best ARIMA({best['p']},{best['d']},{best['q']}) → BIC={best['BIC']:.2f}"
        )

        all_results.append(best)

    pd.DataFrame(all_results).to_csv(metrics_dir / "arima_best_models.csv", index=False)
    print("\nARIMA gridsearch completed → arima_best_models.csv saved")


if __name__ == "__main__":
    arima_gridsearch_from_config()
