"""
ARIMA Model Runner
Fits ARIMA models using best parameters from gridsearch and generates forecasts + metrics.
"""

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

from ..metrics.metrics import mase, smape, mse, rmse, mape, pi_coverage


def run_arima_models(config_path="config.yaml"):
    """
    Run ARIMA models using best parameters from gridsearch.
    Generates forecasts on dev/test sets and computes metrics.
    """
    print("\n=== RUNNING ARIMA MODELS ===\n")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    metrics_dir = Path(cfg["output_metrics"])
    best_params_path = metrics_dir / "arima_best_models.csv"

    if not best_params_path.exists():
        print(f"ERROR: {best_params_path} not found. Run ARIMA gridsearch first!")
        return

    best_params = pd.read_csv(best_params_path)

    forecasts_dir = Path(cfg["output_dir"]) / "forecasts"
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    warm_up = cfg.get("warm_up_days", 60) * 24
    stride = cfg.get("stride_hours", 24)
    horizon = cfg.get("horizon", 24)

    all_metrics = []

    for csv_path in cfg["cleaned_files"]:
        country = Path(csv_path).stem
        print(f"\nProcessing ARIMA for {country}...")

        country_params = best_params[best_params["country"] == country]
        if country_params.empty:
            print(f"  No ARIMA parameters found for {country}, skipping.")
            continue

        p = int(country_params.iloc[0]["p"])
        d = int(country_params.iloc[0]["d"])
        q = int(country_params.iloc[0]["q"])
        print(f"  Using ARIMA({p},{d},{q})")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp").asfreq("1H")

        if df["load"].isnull().any():
            df["load"] = df["load"].interpolate()

        y = df["load"]

        n = len(y)
        split_dev = int(0.8 * n)
        split_test = int(0.9 * n)

        dev = y.iloc[split_dev:split_test]
        test = y.iloc[split_test:]

        dev_index = set(dev.index)
        test_index = set(test.index)

        print(f"  Fitting ARIMA({p},{d},{q}) model...")
        model = SARIMAX(
            y,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, method="lbfgs", maxiter=200)

        steps = list(range(max(warm_up, 1), len(y) - horizon, stride))
        print(f"  Running {len(steps)} forecasting steps...")

        forecasts = []
        for t in tqdm(steps, desc=f"{country} ARIMA"):
            train_end = y.index[t - 1]

            model_t = SARIMAX(
                y.iloc[:t],
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result_t = model_t.fit(
                start_params=result.params,
                disp=False,
                method="lbfgs",
                maxiter=50,
            )

            fcast = result_t.get_forecast(steps=horizon)
            fc_mean = fcast.predicted_mean
            fc_conf = fcast.conf_int(alpha=0.2)

            for i in range(horizon):
                if t + i >= len(y):
                    break

                fc_timestamp = y.index[t + i]

                if fc_timestamp in dev_index:
                    split = "dev"
                elif fc_timestamp in test_index:
                    split = "test"
                else:
                    continue

                forecasts.append(
                    {
                        "split": split,
                        "timestamp": fc_timestamp,
                        "model_type": "ARIMA",
                        "y_true": y.loc[fc_timestamp],
                        "yhat": fc_mean.iloc[i] if i < len(fc_mean) else np.nan,
                        "lo": fc_conf.iloc[i, 0] if i < len(fc_conf) else np.nan,
                        "hi": fc_conf.iloc[i, 1] if i < len(fc_conf) else np.nan,
                        "horizon": i + 1,
                        "train_end": train_end,
                        "country": country,
                    }
                )

        fc = pd.DataFrame(forecasts)

        fc_dev = fc[fc["split"] == "dev"].sort_values("timestamp")
        fc_test = fc[fc["split"] == "test"].sort_values("timestamp")

        fc_dev.to_csv(forecasts_dir / f"{country}_ARIMA_forecasts_dev.csv", index=False)
        fc_test.to_csv(
            forecasts_dir / f"{country}_ARIMA_forecasts_test.csv", index=False
        )

        metrics = []
        for split in ["dev", "test"]:
            sdf = fc[fc["split"] == split]
            if len(sdf) == 0:
                continue

            metrics.append(
                {
                    "country": country,
                    "model": "ARIMA",
                    "split": split,
                    "MASE": mase(sdf.y_true, sdf.yhat, m=24),
                    "sMAPE": smape(sdf.y_true, sdf.yhat),
                    "MSE": mse(sdf.y_true, sdf.yhat),
                    "RMSE": rmse(sdf.y_true, sdf.yhat),
                    "MAPE": mape(sdf.y_true, sdf.yhat),
                    "80_PI_coverage": pi_coverage(sdf.y_true, sdf.lo, sdf.hi),
                }
            )

        pd.DataFrame(metrics).to_csv(
            metrics_dir / f"{country}_ARIMA_metrics.csv", index=False
        )

        all_metrics.extend(metrics)
        print(f"  ✓ {country} ARIMA complete")

    pd.DataFrame(all_metrics).to_csv(
        metrics_dir / "ARIMA_metrics_summary.csv", index=False
    )

    print("\n✓ ARIMA models completed!")
    print(f"  Forecasts saved to: {forecasts_dir}")
    print(f"  Metrics saved to: {metrics_dir}")


def main():
    run_arima_models()


if __name__ == "__main__":
    main()
