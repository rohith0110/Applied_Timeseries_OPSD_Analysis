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
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..metrics.metrics import mase, smape, mse, rmse, mape


from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm


def estimate_seasonal_components(
    y_raw, s, P, Q, seasons_to_use=6, max_sarimax_obs=3 * 168
):
    """
    Robust seasonal extraction and seasonal-AR/MA estimation.

    Returns dict with:
      - seasonal_pattern: length-s array (seasonal means)
      - seasonal_ar_lags: list of last P full-season arrays (each length s)
      - phi_seasonal: scalar seasonal AR estimate
      - theta_seasonal: scalar seasonal MA estimate
      - seasonal_ma_pattern: length-s pattern for MA adjustments
      - residual_seasonal: length-s average residual pattern
      - var_resid_adj: empirical variance of seasonal residuals (used for CI)
    """
    y = np.asarray(y_raw, dtype=np.float64)
    n = len(y)
    out = {}

    if n < s * 2:
        return {
            "seasonal_pattern": np.zeros(s),
            "seasonal_ar_lags": [np.zeros(s) for _ in range(P)],
            "phi_seasonal": 0.0,
            "theta_seasonal": 0.0,
            "seasonal_ma_pattern": np.zeros(s),
            "residual_seasonal": np.zeros(s),
            "var_resid_adj": 0.0,
        }

    try:

        period = s
        window = min(len(y), 3 * s)
        y_small = y[-window:]
        stl = STL(
            y_small,
            period=period,
            robust=False,
            seasonal_deg=0,
            trend_deg=0,
        )
        res = stl.fit()
        seasonal_full = res.seasonal
        trend_full = res.trend
        resid_full = res.resid
    except Exception:

        n_seasons = n // s
        seasons = []
        for k in range(n_seasons):
            block = y[k * s : (k + 1) * s]
            if len(block) == s:
                seasons.append(block)
        if len(seasons) > 0:
            seasonal_full = np.tile(np.mean(seasons, axis=0), n_seasons)[:n]
            trend_full = np.convolve(
                y,
                np.ones(min(s, max(3, n // 10))) / min(s, max(3, n // 10)),
                mode="same",
            )
            resid_full = y - seasonal_full - trend_full
        else:
            seasonal_full = np.zeros(n)
            trend_full = np.zeros(n)
            resid_full = y - np.mean(y)

    trend_recent = trend_full[-min(len(trend_full), s * 4) :]
    trend_slope = (trend_recent[-1] - trend_recent[0]) / len(trend_recent)

    n_seasons = n // s
    use_seasons = min(n_seasons, seasons_to_use)
    seasonal_blocks = []
    for k in range(n_seasons - use_seasons, n_seasons):
        block = y[k * s : (k + 1) * s]
        if len(block) == s:
            seasonal_blocks.append(block - np.mean(block))
    if len(seasonal_blocks) == 0:
        seasonal_pattern = np.zeros(s)
    else:
        seasonal_pattern = np.mean(seasonal_blocks, axis=0)

    seasonal_ar_lags = []
    for lag_idx in range(1, P + 1):
        idx = -lag_idx * s
        if abs(idx) <= len(y) - s:
            seasonal_ar_lags.append(y[idx : idx + s].copy())
        else:
            seasonal_ar_lags.append(seasonal_pattern.copy())

    phi_seasonal = 0.0
    try:

        X_seasons = []
        Y_seasons = []
        for k in range(1, n_seasons):
            prev = y[(k - 1) * s : k * s]
            curr = y[k * s : (k + 1) * s]
            if len(prev) == s and len(curr) == s:
                X_seasons.append(prev - np.mean(prev))
                Y_seasons.append(curr - np.mean(curr))
        if len(X_seasons) >= 2:
            Xm = np.vstack(X_seasons)
            Ym = np.vstack(Y_seasons)

            X_flat = Xm.flatten()
            Y_flat = Ym.flatten()

            X_design = sm.add_constant(X_flat)
            ols = sm.OLS(Y_flat, X_design).fit()
            phi_hat = ols.params[1]
            phi_seasonal = float(np.clip(phi_hat, -0.95, 0.95))
        else:
            phi_seasonal = 0.0
    except Exception:
        phi_seasonal = 0.0

    theta_seasonal = None
    try:

        window = min(len(y), max_sarimax_obs)
        y_window = y[-window:]

        sm_order = (0, 0, 0)
        sm_seasonal = (1, 0, 1, s)
        sm_mod = SARIMAX(
            y_window,
            order=sm_order,
            seasonal_order=sm_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sm_res = sm_mod.fit(disp=False, maxiter=50)
        params = dict(zip(sm_res.param_names, sm_res.params))
        theta = params.get("seasonal_ma.L1", None)
        if theta is not None and not np.isnan(theta):
            theta_seasonal = float(np.clip(theta, -0.95, 0.95))
    except Exception:
        theta_seasonal = None

    if theta_seasonal is None:

        try:
            if n >= 3 * s:
                recent = y[-3 * s :]
                diffs = []
                for i in range(len(recent) - s):
                    diffs.append(recent[i + s] - recent[i])
                diffs = np.array(diffs)
                if len(diffs) > s:
                    acf_s = np.corrcoef(diffs[:-s], diffs[s:])[0, 1]
                    theta_seasonal = float(np.clip(-acf_s * 0.7, -0.85, 0.85))
                else:
                    theta_seasonal = -0.4
            else:
                theta_seasonal = -0.4
        except Exception:
            theta_seasonal = -0.4

    resid = np.asarray(resid_full, dtype=np.float64)

    residual_seasonal = np.zeros(s)
    counts = np.zeros(s)
    for idx in range(len(resid) - s + 1):
        pos = idx % s
        residual_seasonal[pos] += resid[idx]
        counts[pos] += 1
    counts[counts == 0] = 1
    residual_seasonal = residual_seasonal / counts

    seasonal_ma_pattern = residual_seasonal * (
        theta_seasonal if theta_seasonal is not None else -0.4
    )

    var_resid_adj = np.var(
        resid.reshape(-1, 1)
        if len(resid) < s
        else resid[-(use_seasons * s) :].reshape(-1, 1)
    )

    return {
        "seasonal_pattern": seasonal_pattern.astype(np.float64),
        "seasonal_ar_lags": seasonal_ar_lags,
        "phi_seasonal": float(phi_seasonal),
        "theta_seasonal": float(theta_seasonal),
        "seasonal_ma_pattern": seasonal_ma_pattern.astype(np.float64),
        "residual_seasonal": residual_seasonal.astype(np.float64),
        "var_resid_adj": float(np.var(resid)),
        "trend_slope": float(trend_slope),
    }


CALENDAR_FEATURE_COLUMNS = [
    "bias",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]


def build_calendar_features(index):
    """Return deterministic calendar signals (bias + harmonics)."""
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    bias = np.ones(len(index), dtype=np.float32)
    sin_hour = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    cos_hour = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
    sin_dow = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    cos_dow = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    return np.column_stack([bias, sin_hour, cos_hour, sin_dow, cos_dow])


def build_transformed_exog(exog_raw, s):
    """Apply the same seasonal+regular differencing used on load."""
    if exog_raw is None or len(exog_raw) <= s + 1:
        return None
    seasonal = exog_raw[s:, :] - exog_raw[:-s, :]
    transformed = seasonal[1:, :] - seasonal[:-1, :]
    return transformed.astype(np.float32)


def build_future_transformed_exog(exog_hist, exog_future, s):
    """Create future exog design matrix aligned with transformed series."""
    if (
        exog_hist is None
        or exog_future is None
        or len(exog_hist) <= s + 1
        or len(exog_future) == 0
    ):
        return None
    combined = np.vstack([exog_hist, exog_future])
    seasonal = combined[s:, :] - combined[:-s, :]
    transformed = seasonal[1:, :] - seasonal[:-1, :]
    return transformed[-len(exog_future) :, :].astype(np.float32)


def _unpack_forecast_result(result):
    conf = None
    if isinstance(result, tuple):
        z_pred = result[0]
        if len(result) == 2:
            conf = result[1]
        elif len(result) >= 3:
            lower = result[1]
            upper = result[2]
            conf = (lower, upper)
    else:
        z_pred = result
    return z_pred, conf


def forecast_with_conf_int(model, steps, level=0.8, exog=None):
    """Call cuML forecast, attempting to retrieve confidence intervals."""
    conf = None
    try:
        result = model.forecast(steps, level=level, exog=exog, return_conf_int=True)
        z_pred, conf = _unpack_forecast_result(result)
    except TypeError:
        result = model.forecast(steps, level=level, exog=exog)
        z_pred, conf = _unpack_forecast_result(result)
    except Exception:
        z_pred = model.forecast(steps)

    if isinstance(z_pred, cudf.Series):
        z_pred = z_pred.to_numpy()
    elif hasattr(z_pred, "to_numpy"):
        z_pred = np.asarray(z_pred.to_numpy())
    else:
        z_pred = np.asarray(z_pred)

    if conf is not None:
        if isinstance(conf, tuple) and len(conf) >= 2:
            lower_raw, upper_raw = conf[0], conf[1]
            if hasattr(lower_raw, "to_numpy"):
                lower_raw = lower_raw.to_numpy()
            if hasattr(upper_raw, "to_numpy"):
                upper_raw = upper_raw.to_numpy()
            conf = np.column_stack([np.asarray(lower_raw), np.asarray(upper_raw)])
        elif isinstance(conf, cudf.DataFrame):
            conf = conf.to_pandas().to_numpy()
        elif hasattr(conf, "to_numpy"):
            conf = conf.to_numpy()
        else:
            conf = np.asarray(conf)
    return z_pred, conf


def estimate_theta_from_statsmodels(
    y_raw, exog_raw, order, seasonal_order, exog_columns, max_obs=2000
):
    """Fit a small SARIMAX on CPU to estimate seasonal MA coefficient."""
    try:
        subset_y = y_raw[-max_obs:]
        subset_exog = (
            pd.DataFrame(exog_raw[-max_obs:], columns=exog_columns)
            if exog_raw is not None
            else None
        )
        has_constant = False
        if subset_exog is not None:
            try:
                first_col = subset_exog.iloc[:, 0].to_numpy()
                if (
                    np.allclose(first_col, first_col[0])
                    and abs(first_col[0] - 1.0) < 1e-6
                ):
                    has_constant = True
            except Exception:
                has_constant = False
        trend = "n" if has_constant else "c"

        model = SARIMAX(
            subset_y,
            order=order,
            seasonal_order=seasonal_order,
            exog=subset_exog,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=100)
        params = dict(zip(res.param_names, res.params))
        theta = params.get("seasonal_ma.L1", np.nan)
        if theta is not None and not np.isnan(theta):
            return float(np.clip(theta, -0.95, 0.95))
    except Exception as exc:
        print(f"⚠ Statsmodels seasonal MA estimation failed: {exc}")
    return None


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


def two_stage_fit(y_raw, s, P, Q, p, q, d, D, exog_raw=None, exog_columns=None):
    """
    Enhanced SARIMA(p,d,q)x(P,D,Q,s) using:
      1) Seasonal+regular differencing for cuML ARIMA
      2) cuML ARIMA(p,0,q) on GPU
      3) Robust seasonal decomposition + seasonal AR/MA estimation
    """

    z = build_transformed_target(y_raw, s)

    z_cu = cudf.Series(z.astype(np.float32))

    if exog_raw is not None:
        transformed_exog = build_transformed_exog(exog_raw, s)
        if transformed_exog is not None:
            exog_cu = cudf.DataFrame(transformed_exog, columns=exog_columns)
        else:
            exog_cu = None
    else:
        exog_cu = None

    model = cuARIMA(z_cu, order=(p, 0, q), exog=exog_cu)
    model.fit()

    seasonal_info = estimate_seasonal_components(
        y_raw=y_raw,
        s=s,
        P=P,
        Q=Q,
        seasons_to_use=6,
        max_sarimax_obs=3 * s * 7,
    )

    seasonal_info.update(
        {
            "P": P,
            "Q": Q,
            "s": s,
            "exog_columns": exog_columns,
            "exog_hist": exog_raw if exog_raw is not None else None,
            "last_values": y_raw[-s:].copy() if len(y_raw) >= s else y_raw.copy(),
            "recent_mean": np.mean(y_raw[-s:]) if len(y_raw) >= s else np.mean(y_raw),
        }
    )

    return model, seasonal_info


def forecast_horizon(
    model,
    y_train_raw,
    seasonal_info,
    H,
    s,
    exog_hist=None,
    exog_future=None,
    level=0.8,
):
    """
    Enhanced SARIMA forecasting with P and Q components:
      1) Get ARIMA(p,0,q) forecast on differenced series
      2) Invert differencing
      3) Apply Seasonal AR (P) adjustment - mimics Φ_P(B^s)
      4) Apply Seasonal MA (Q) adjustment - mimics Θ_Q(B^s)
      5) Apply trend and level corrections
    """

    future_exog_cu = None
    if exog_future is not None and exog_hist is not None:
        transformed_future = build_future_transformed_exog(exog_hist, exog_future, s)
        if transformed_future is not None:
            cols = seasonal_info.get("exog_columns") or CALENDAR_FEATURE_COLUMNS
            future_exog_cu = cudf.DataFrame(transformed_future, columns=cols)

    z_pred, conf = forecast_with_conf_int(model, H, level=level, exog=future_exog_cu)
    z_pred = z_pred.astype(np.float64)

    y_pred = invert_transforms_exact(z_pred, y_train_raw, s)
    if conf is not None and conf.shape[1] >= 2:
        lower = invert_transforms_exact(conf[:, 0], y_train_raw, s)
        upper = invert_transforms_exact(conf[:, 1], y_train_raw, s)
    else:
        lower = np.full_like(y_pred, np.nan)
        upper = np.full_like(y_pred, np.nan)

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
        if not np.isnan(lower[h]):
            lower[h] += total_adjustment
        if not np.isnan(upper[h]):
            upper[h] += total_adjustment

    return y_pred, lower, upper


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

    calendar_feats = build_calendar_features(df.index)

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
        exog_train = calendar_feats[:t]

        if len(y_train) > max_train_size:
            y_train = y_train[-max_train_size:]
            exog_train = exog_train[-max_train_size:]

        if (step_idx % refit_interval == 0) or (cached_model is None):

            free_gb = gpu_check(1.0)
            if free_gb < 1.0:
                print("⚠ Low VRAM — skipping this step")
                continue

            cached_model, cached_seasonal_info = two_stage_fit(
                y_raw=y_train,
                s=s,
                P=P,
                Q=Q,
                p=p,
                q=q,
                d=d,
                D=D,
                exog_raw=exog_train,
                exog_columns=CALENDAR_FEATURE_COLUMNS,
            )

        exog_future = calendar_feats[t : t + horizon]
        yhat, lo_vals, hi_vals = forecast_horizon(
            cached_model,
            y_train,
            cached_seasonal_info,
            horizon,
            s,
            exog_hist=exog_train,
            exog_future=exog_future,
        )

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
            lo = float(lo_vals[i]) if not np.isnan(lo_vals[i]) else np.nan
            hi = float(hi_vals[i]) if not np.isnan(hi_vals[i]) else np.nan

            records.append(
                {
                    "split": split,
                    "timestamp": ts,
                    "model_type": "SARIMA_FOURIER_FAST",
                    "y_true": y_true,
                    "yhat": float(yhat[i]),
                    "lo": lo,
                    "hi": hi,
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

        mask = sdf["lo"].notna() & sdf["hi"].notna()
        coverage = (
            float(
                np.mean(
                    (sdf.loc[mask, "y_true"] >= sdf.loc[mask, "lo"])
                    & (sdf.loc[mask, "y_true"] <= sdf.loc[mask, "hi"])
                )
            )
            if mask.any()
            else np.nan
        )

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
                "80_PI_coverage": coverage,
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
