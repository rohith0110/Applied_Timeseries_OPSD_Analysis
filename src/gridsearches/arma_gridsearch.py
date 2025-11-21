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


def fit_single_arma(params, y, country):
    """Fit a single ARMA model on differenced stationary data."""
    p, q = params
    import time

    start = time.time()
    try:
        model = SARIMAX(
            y,
            order=(p, 0, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, method="lbfgs", maxiter=100, concentrate_scale=True)
        elapsed = time.time() - start
        print(f"[{country}] ✓ ARMA({p},{q}) finished in {elapsed:.1f}s")

        return {"country": country, "p": p, "q": q, "AIC": model.aic, "BIC": model.bic}
    except Exception:
        return None


def arma_gridsearch_from_config(
    config_path="config.yaml", use_parallel=True, subsample_data=True
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(cfg["output_metrics"])
    grid_dir = metrics_dir / "gridsearch"
    grid_dir.mkdir(parents=True, exist_ok=True)

    arma_cfg = cfg["arma"]
    p_range = arma_cfg["p_range"]
    q_range = arma_cfg["q_range"]

    all_results = []

    for csv_path in cfg["cleaned_files"]:
        country = Path(csv_path).stem
        df = (
            pd.read_csv(csv_path, parse_dates=["timestamp"])
            .set_index("timestamp")
            .asfreq("h")
        )

        if df["load"].isnull().any():
            print(f"   -> {country}: interpolating missing timestamps")
            df["load"] = df["load"].interpolate()

        y = df["load"]

        y = y.diff().dropna()

        if subsample_data and len(y) > 17280:
            print(f"   -> Using last 6 months (17280 points) for {country}")
            y = y.iloc[-17280:]

        print(f"\nRunning ARMA gridsearch for {country}...")

        param_combos = list(product(p_range, q_range))
        results = []

        if use_parallel:
            n_cores = max(1, mp.cpu_count() - 1)
            print(f"   -> Using {n_cores} CPU cores")

            fit_func = partial(fit_single_arma, y=y, country=country)

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
                r = fit_single_arma(params, y, country)
                if r:
                    results.append(r)

        if not results:
            print(f"No valid ARMA fits for {country}")
            continue

        df_res = pd.DataFrame(results).sort_values("BIC").reset_index(drop=True)
        df_res.to_csv(grid_dir / f"{country}_arma_grid.csv", index=False)

        best = df_res.iloc[0].to_dict()
        print(f"Best ARMA({best['p']},{best['q']})  → BIC={best['BIC']:.2f}")

        all_results.append(best)

    pd.DataFrame(all_results).to_csv(metrics_dir / "arma_best_models.csv", index=False)
    print("\nARMA gridsearch completed → arma_best_models.csv saved")


if __name__ == "__main__":
    arma_gridsearch_from_config()
