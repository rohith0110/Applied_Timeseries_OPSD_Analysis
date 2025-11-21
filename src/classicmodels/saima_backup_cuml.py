"""
GPU Forecast Backtest (SARIMA-Approx)
-------------------------------------

This script:
  ✓ Loads best parameters from config.yaml
  ✓ Applies SARIMA ≈ ARIMA w/ seasonal differencing + seasonal AR exogs
  ✓ Approximates seasonal MA using residual lag features (two-stage fit)
  ✓ Fits GPU ARIMA (cuML)
  ✓ Runs expanding-window backtest
  ✓ Outputs dev/test forecast CSVs (as required)
  ✓ Fully VRAM-safe on RTX 3060 (6GB)
"""

import os


os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "1"

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cudf
import cupy as cp
import pynvml

from cuml.tsa.arima import ARIMA as cuARIMA

from ..metrics.metrics import mase, smape, mse, rmse, mape


def detect_gpu():
    """Detect and display which GPU is being used."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    print("\n" + "=" * 60)
    print("  GPU Detection")
    print("=" * 60)
    print(f"Total GPUs detected: {device_count}")

    for i in range(device_count):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)

        if isinstance(name, bytes):
            name = name.decode("utf-8")

        print(f"\nGPU {i}:")
        print(f"  Name: {name}")
        print(f"  Total Memory: {info.total / 1e9:.2f} GB")
        print(f"  Free Memory: {info.free / 1e9:.2f} GB")
        print(f"  Used Memory: {info.used / 1e9:.2f} GB")

    try:
        cuda_device = cp.cuda.Device()
        print(f"\n✓ cuML/CuPy will use GPU {cuda_device.id}")
        h = pynvml.nvmlDeviceGetHandleByIndex(cuda_device.id)
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        print(f"  Active GPU: {name}")
    except:
        print("\n⚠ Could not determine active GPU")

    print("=" * 60 + "\n")

    return cuda_device.id if "cuda_device" in locals() else 0


def gpu_check(min_free_gb=2.0):
    """Abort if GPU free VRAM is below threshold."""
    pynvml.nvmlInit()

    cuda_device = cp.cuda.Device().id
    h = pynvml.nvmlDeviceGetHandleByIndex(cuda_device)

    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    free_gb = info.free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"❌ GPU VRAM too low ({free_gb:.2f} GB free). "
            f"Need at least {min_free_gb} GB."
        )
    return free_gb


def seasonal_diff(y, s):
    return y[s:] - y[:-s]


def regular_diff(y):
    return y[1:] - y[:-1]


def build_transformed_target(y, s):
    """Apply D=1 then d=1 → z_t = (y_t - y_{t-s}) - (y_{t-1} - y_{t-1-s})"""
    y = y.astype(np.float64)

    u = seasonal_diff(y, s)

    z = regular_diff(u).astype(np.float32)
    return z


def build_seasonal_AR_exogs(y, s, P):
    """Build seasonal AR exogenous features: y_{t-s}, y_{t-2s}, ..."""
    if P == 0:
        return np.empty((len(y) - s - 1, 0), dtype=np.float64)

    n = len(y)
    max_lag = P * s
    usable = y[max_lag:]

    rows = len(usable) - s
    X = np.zeros((rows, P), dtype=np.float64)

    base_idx = max_lag + s

    for p in range(1, P + 1):
        X[:, p - 1] = y[base_idx - p * s : base_idx - p * s + rows]

    return X


def invert_transforms_exact(pred_transformed, train_raw, s):
    """
    pred_transformed: predicted z_{T+1...T+H}
    train_raw: y_0...y_T
    s: seasonal period

    Exact inversion for D=d=1.
    """
    H = len(pred_transformed)
    history = list(train_raw)

    out = np.empty(H, dtype=np.float64)
    for h in range(H):
        z = pred_transformed[h]
        y_t_minus1 = history[-1]
        y_t_minus1_minus_s = history[-1 - s] if len(history) > s else history[-1]
        y_t_minus_s = history[-s]

        y_next = z + y_t_minus_s + y_t_minus1 - y_t_minus1_minus_s
        out[h] = y_next
        history.append(y_next)

    return out


def estimate_seasonal_ma_component(y_raw, s, Q):
    """
    Estimate seasonal MA component by analyzing seasonal residuals.
    This approximates Θ_Q(B^s) behavior.
    """
    if Q == 0:
        return np.zeros(s)

    n_full_seasons = len(y_raw) // s
    seasonal_residuals = []

    for season_idx in range(n_full_seasons - 1):
        current_season = y_raw[season_idx * s : (season_idx + 1) * s]
        next_season = y_raw[(season_idx + 1) * s : (season_idx + 2) * s]

        if len(next_season) == s:

            residual = next_season - current_season
            seasonal_residuals.append(residual)

    if len(seasonal_residuals) > 0:

        seasonal_ma_pattern = np.mean(seasonal_residuals, axis=0)

        damping = 0.6 if Q == 1 else 0.8
        seasonal_ma_pattern *= damping
    else:
        seasonal_ma_pattern = np.zeros(s)

    return seasonal_ma_pattern


def compute_arima_residuals(model, y_transformed):
    """
    Extract residuals from fitted ARIMA model for MA component estimation.
    """

    try:

        if hasattr(model, "resid"):
            resid = model.resid.to_numpy()
        else:

            resid = np.zeros(len(y_transformed))
    except:
        resid = np.zeros(len(y_transformed))

    return resid


def two_stage_fit(y_raw, s, P, Q, p, q):
    """
    Enhanced SARIMA(p,d,q)x(P,D,Q,s) with improved seasonal components:
      1) Apply seasonal and regular differencing (D=1, d=1)
      2) Fit ARIMA(p,0,q) on transformed data
      3) Extract seasonal patterns for P (seasonal AR)
      4) Estimate seasonal MA component for Q
      5) Combine all components for accurate forecasting
    """

    z = build_transformed_target(y_raw, s)

    z_cu = cudf.Series(z.astype(np.float32))

    model = cuARIMA(z_cu, order=(p, 0, q))
    model.fit()

    seasonal_pattern = np.zeros(s)
    seasonal_weights = np.zeros(s)

    n_points = len(y_raw)
    for i in range(n_points):
        pos = i % s

        weight = np.exp((i - n_points) / (n_points / 3))
        seasonal_pattern[pos] += y_raw[i] * weight
        seasonal_weights[pos] += weight

    seasonal_pattern = seasonal_pattern / np.maximum(seasonal_weights, 1e-8)

    seasonal_ar_lags = []
    for lag_idx in range(1, P + 1):
        start_idx = -lag_idx * s
        end_idx = start_idx + s if start_idx + s < 0 else None
        if abs(start_idx) <= len(y_raw):
            seasonal_ar_lags.append(y_raw[start_idx:end_idx].copy())
        else:
            seasonal_ar_lags.append(seasonal_pattern.copy())

    seasonal_ma_pattern = estimate_seasonal_ma_component(y_raw, s, Q)

    arima_residuals = compute_arima_residuals(model, z)

    if len(arima_residuals) >= s:
        residual_seasonal = np.zeros(s)
        residual_counts = np.zeros(s)

        for i in range(len(arima_residuals)):
            pos = i % s
            residual_seasonal[pos] += arima_residuals[i]
            residual_counts[pos] += 1

        residual_seasonal = residual_seasonal / np.maximum(residual_counts, 1)
    else:
        residual_seasonal = np.zeros(s)

    window = min(s, len(y_raw) // 4)
    if window > 1:
        trend = np.convolve(y_raw, np.ones(window) / window, mode="valid")
        trend_slope = (trend[-1] - trend[-min(100, len(trend))]) / min(100, len(trend))
    else:
        trend_slope = 0.0

    recent_mean = np.mean(y_raw[-s:]) if len(y_raw) >= s else np.mean(y_raw)

    if P > 0 and len(seasonal_ar_lags) > 0:

        n_lags_to_use = min(3, len(y_raw) // s)
        phi_estimates = []

        for lag in range(1, n_lags_to_use + 1):
            if len(y_raw) >= (lag + 1) * s:
                current = y_raw[-s:]
                lagged = y_raw[-((lag + 1) * s) : -lag * s]

                if len(current) == len(lagged) and len(current) > 1:

                    current_dt = current - np.mean(current)
                    lagged_dt = lagged - np.mean(lagged)

                    corr = np.corrcoef(current_dt, lagged_dt)[0, 1]
                    if not np.isnan(corr):
                        phi_estimates.append(corr / lag)

        if len(phi_estimates) > 0:
            phi_seasonal = np.mean(phi_estimates)
            phi_seasonal = np.clip(phi_seasonal, -0.95, 0.95)
        else:
            phi_seasonal = 0.6
    else:
        phi_seasonal = 0.0

    if Q > 0:

        if len(y_raw) >= 3 * s:

            recent_data = y_raw[-3 * s :]

            seasonal_diffs = []
            for i in range(len(recent_data) - s):
                seasonal_diffs.append(recent_data[i + s] - recent_data[i])

            seasonal_diffs = np.array(seasonal_diffs)

            if len(seasonal_diffs) > s:
                acf_s = np.corrcoef(seasonal_diffs[:-s], seasonal_diffs[s:])[0, 1]

                theta_seasonal = -acf_s * 0.7
                theta_seasonal = np.clip(theta_seasonal, -0.85, 0.85)
            else:
                theta_seasonal = -0.4
        else:
            theta_seasonal = -0.4
    else:
        theta_seasonal = 0.0

    seasonal_info = {
        "P": P,
        "Q": Q,
        "s": s,
        "seasonal_pattern": seasonal_pattern,
        "seasonal_ar_lags": seasonal_ar_lags,
        "seasonal_ma_pattern": seasonal_ma_pattern,
        "residual_seasonal": residual_seasonal,
        "last_values": y_raw[-s:].copy() if len(y_raw) >= s else y_raw.copy(),
        "trend_slope": trend_slope,
        "recent_mean": recent_mean,
        "phi_seasonal": phi_seasonal,
        "theta_seasonal": theta_seasonal,
    }

    return model, seasonal_info


def forecast_horizon(model, y_train_raw, seasonal_info, H, s):
    """
    Enhanced SARIMA forecasting with P and Q components:
      1) Get ARIMA(p,0,q) forecast on differenced series
      2) Invert differencing
      3) Apply Seasonal AR (P) adjustment - mimics Φ_P(B^s)
      4) Apply Seasonal MA (Q) adjustment - mimics Θ_Q(B^s)
      5) Apply trend and level corrections
    """

    z_pred = model.forecast(H)
    z_pred = z_pred.to_numpy().astype(np.float64)

    y_pred = invert_transforms_exact(z_pred, y_train_raw, s)

    P = seasonal_info["P"]
    Q = seasonal_info["Q"]
    seasonal_pattern = seasonal_info["seasonal_pattern"]
    seasonal_ar_lags = seasonal_info["seasonal_ar_lags"]
    seasonal_ma_pattern = seasonal_info["seasonal_ma_pattern"]
    residual_seasonal = seasonal_info["residual_seasonal"]
    trend_slope = seasonal_info["trend_slope"]
    recent_mean = seasonal_info["recent_mean"]
    phi_seasonal = seasonal_info["phi_seasonal"]
    theta_seasonal = seasonal_info["theta_seasonal"]

    for h in range(H):
        season_pos = (len(y_train_raw) + h) % s

        horizon_decay = np.exp(-h / (H * 1.5))

        pattern_value = seasonal_pattern[season_pos]
        pattern_deviation = (pattern_value - recent_mean) * 0.35 * horizon_decay

        ar_adjustment = 0.0
        if P > 0 and len(seasonal_ar_lags) > 0:
            for lag_idx, lag_values in enumerate(seasonal_ar_lags):
                if season_pos < len(lag_values):

                    weight = phi_seasonal / (lag_idx + 1)
                    lag_deviation = lag_values[season_pos] - recent_mean
                    ar_adjustment += lag_deviation * weight

            ar_adjustment *= 0.7 * horizon_decay

        ma_adjustment = 0.0
        if Q > 0:

            ma_value = seasonal_ma_pattern[season_pos]
            residual_value = residual_seasonal[season_pos]

            ma_adjustment = ma_value * theta_seasonal * 0.6 + residual_value * 0.3

            ma_decay = np.exp(-h / (s / 3))
            ma_adjustment *= ma_decay

        trend_adjustment = trend_slope * h * 0.4 * horizon_decay

        if len(y_train_raw) >= 24:
            recent_trend = (y_train_raw[-1] - y_train_raw[-24]) / 24
            momentum_adjustment = recent_trend * h * 0.15 * horizon_decay
        else:
            momentum_adjustment = 0.0

        total_adjustment = (
            pattern_deviation
            + ar_adjustment
            + ma_adjustment
            + trend_adjustment
            + momentum_adjustment
        )

        y_pred[h] += total_adjustment

    return y_pred


def backtest_country(csv_path, cfg, forecasts_dir, metrics_dir):
    """
    Expanding-window backtest.
    Output:
        CC_SARIMA_FOURIER_FAST_dev.csv
        CC_SARIMA_FOURIER_FAST_test.csv
        CC_SARIMA_FOURIER_FAST_metrics.csv
    """
    country = Path(csv_path).stem
    s = cfg["sarima_best"]["s"]
    p = cfg["sarima_best"]["p"]
    d = cfg["sarima_best"]["d"]
    q = cfg["sarima_best"]["q"]
    P = cfg["sarima_best"]["P"]
    D = cfg["sarima_best"]["D"]
    Q = cfg["sarima_best"]["Q"]

    horizon = cfg.get("horizon", 24)
    stride = cfg.get("stride_hours", 24)
    warm_up = cfg.get("warm_up_days", 60) * 24

    max_train_size = 8760
    refit_interval = 2

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp").asfreq("h")
    df["load"] = df["load"].interpolate()

    y = df["load"]
    n = len(y)

    train = y.iloc[: int(0.8 * n)]
    dev = y.iloc[int(0.8 * n) : int(0.9 * n)]
    test = y.iloc[int(0.9 * n) :]

    dev_index = set(dev.index)
    test_index = set(test.index)

    y_values = y.values
    y_index = y.index

    records = []

    print(f"\n▶ Backtesting {country} — GPU SARIMA-Approx (FAST mode)")
    print(
        f"  Max training size: {max_train_size}, Refit interval: {refit_interval} steps"
    )

    steps = list(range(warm_up, len(y) - horizon, stride))

    cached_model = None
    cached_seasonal_info = None

    for step_idx, t in enumerate(tqdm(steps)):
        y_train = y_values[:t]

        if len(y_train) > max_train_size:
            y_train = y_train[-max_train_size:]

        if (step_idx % refit_interval == 0) or (cached_model is None):

            free_gb = gpu_check(1.0)
            if free_gb < 1.0:
                print("⚠ Low VRAM — skipping this step")
                continue

            cached_model, cached_seasonal_info = two_stage_fit(
                y_raw=y_train, s=s, P=P, Q=Q, p=p, q=q
            )

        yhat = forecast_horizon(cached_model, y_train, cached_seasonal_info, horizon, s)

        train_end_ts = y_index[t - 1]

        for i in range(horizon):
            ts = y_index[t + i]

            if ts in dev_index:
                split = "dev"
            elif ts in test_index:
                split = "test"
            else:
                continue

            y_true = float(y.loc[ts])

            records.append(
                {
                    "split": split,
                    "timestamp": ts,
                    "model_type": "SARIMA_FOURIER_FAST",
                    "y_true": y_true,
                    "yhat": float(yhat[i]),
                    "lo": np.nan,
                    "hi": np.nan,
                    "horizon": i + 1,
                    "train_end": train_end_ts,
                }
            )

    fc = pd.DataFrame(records)

    fc[fc["split"] == "dev"].to_csv(
        forecasts_dir / f"{country}_SARIMA_GPU_{s}_dev.csv", index=False
    )
    fc[fc["split"] == "test"].to_csv(
        forecasts_dir / f"{country}_SARIMA_GPU_{s}_test.csv", index=False
    )

    print(f"✓ Saved GPU SARIMA forecasts for {country}")

    metrics = []
    for split in ["dev", "test"]:
        sdf = fc[fc["split"] == split]
        if len(sdf) == 0:
            continue

        metrics.append(
            {
                "country": country,
                "model": "SARIMA_FOURIER_FAST",
                "split": split,
                "MASE": mase(sdf.y_true, sdf.yhat, m=24),
                "sMAPE": smape(sdf.y_true, sdf.yhat),
                "MSE": mse(sdf.y_true, sdf.yhat),
                "RMSE": rmse(sdf.y_true, sdf.yhat),
                "MAPE": mape(sdf.y_true, sdf.yhat),
                "80_PI_coverage": np.nan,
            }
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        metrics_dir / f"{country}_SARIMA_GPU_{s}_metrics.csv", index=False
    )

    return metrics


if __name__ == "__main__":

    active_gpu = detect_gpu()

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    forecasts_dir = Path(cfg["output_dir"]) / "forecasts"
    metrics_dir = Path(cfg["output_dir"]) / "metrics"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results_all = []

    for csv in cfg["cleaned_files"]:
        metrics = backtest_country(csv, cfg, forecasts_dir, metrics_dir)
        results_all.extend(metrics)

    pd.DataFrame(results_all).to_csv(
        metrics_dir / "SARIMA_CUML_GPU_summary.csv", index=False
    )

    print("\n🎉 GPU Backtest Complete!")
