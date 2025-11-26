import os
import warnings
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
import glob

from exogenous_features import build_exogenous, build_future_exogenous

warnings.filterwarnings("ignore")
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")


def _log(msg: str):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


def _read_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_cleaned_country_files(data_folder: str = "data") -> Dict[str, pd.DataFrame]:
    """
    Load already cleaned country CSV files from the data folder.
    Expects files named like: {country_code}_cleaned.csv or {country_code}.csv
    """
    dfs_by_cc = {}

    # Get the project root directory (2 levels up from this file)
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(root_dir, data_folder)

    _log(f"Looking for data files in: {data_path}")

    if not os.path.exists(data_path):
        _log(f"ERROR: data folder not found at {data_path}")
        return dfs_by_cc

    # Find all CSV files (try both patterns)
    csv_files = []
    for pattern in ["*_cleaned.csv", "*.csv"]:
        csv_files.extend(glob.glob(os.path.join(data_path, pattern)))

    # Remove duplicates
    csv_files = list(set(csv_files))

    _log(f"Found {len(csv_files)} CSV files in {data_path}")

    for filepath in csv_files:
        filename = os.path.basename(filepath)

        # Skip non-country files
        if filename.startswith(".") or "test" in filename.lower():
            continue

        # Extract country code from filename
        # Handles: "DE_cleaned.csv" -> "DE", "DE.csv" -> "DE"
        country_code = filename.replace("_cleaned.csv", "").replace(".csv", "")

        # Skip if already loaded (prefer _cleaned version)
        if country_code in dfs_by_cc:
            continue

        try:
            _log(f"Loading {filename}...")
            df = pd.read_csv(filepath)

            # Ensure timestamp column exists and is datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            elif "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])
            else:
                _log(f"Warning: No timestamp column found in {filename}, skipping")
                continue

            # Ensure load column exists
            if "load" not in df.columns:
                _log(f"Warning: No 'load' column found in {filename}, skipping")
                continue

            dfs_by_cc[country_code] = df
            _log(
                f"✓ Loaded {country_code}: {len(df)} rows, columns: {list(df.columns)}"
            )

        except Exception as e:
            _log(f"ERROR loading {filename}: {e}")

    if not dfs_by_cc:
        _log("WARNING: No valid country data files were loaded!")

    return dfs_by_cc


def select_sarima_order(
    y: pd.Series, X: Optional[pd.DataFrame], sarima_cfg: dict
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    p_range = sarima_cfg.get("p_range", [0, 1, 2])
    d_range = sarima_cfg.get("d_range", [0, 1])
    q_range = sarima_cfg.get("q_range", [0, 1, 2])
    P_range = sarima_cfg.get("P_range", [0, 1])
    D_range = sarima_cfg.get("D_range", [0, 1])
    Q_range = sarima_cfg.get("Q_range", [0, 1])
    s = sarima_cfg.get("s", 168)

    best = None
    best_metrics = (np.inf, np.inf)  # (BIC, AIC)
    total = (
        len(p_range)
        * len(d_range)
        * len(q_range)
        * len(P_range)
        * len(D_range)
        * len(Q_range)
    )
    count = 0
    interval = max(1, total // 5)
    _log(f"Order search: {total} combinations to evaluate (grid from config)")
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            try:
                                model = SARIMAX(
                                    y,
                                    exog=X,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                res = model.fit(disp=False)
                                bic, aic = res.bic, res.aic
                                if (bic < best_metrics[0]) or (
                                    np.isclose(bic, best_metrics[0])
                                    and aic < best_metrics[1]
                                ):
                                    best_metrics = (bic, aic)
                                    best = (order, seasonal_order)
                            except Exception:
                                continue
                            finally:
                                count += 1
                                if count % interval == 0 or count == total:
                                    msg = f"Order search progress: {count}/{total}"
                                    if best is not None:
                                        msg += f" | current best order={best[0]}, seasonal={best[1]} (BIC={best_metrics[0]:.1f})"
                                    _log(msg)
    if best is None:
        best = ((1, 1, 0), (1, 1, 0, s))
    return best


def expanding_backtest(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    config: dict,
    include_calendar: bool,
    include_wind: bool,
    include_solar: bool,
    cc: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if "timestamp" in df.columns:
        df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    y = df["load"].astype(float)

    ratios = config["forecasting"]
    train_ratio = float(ratios.get("train_ratio", 0.8))
    val_ratio = float(ratios.get("val_ratio", 0.1))
    test_ratio = float(ratios.get("test_ratio", 0.1))

    n = len(y)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    idx = y.index
    train_end_ts = idx[n_train - 1]
    dev_end_ts = idx[n_train + n_val - 1]

    horizon = int(ratios.get("horizon", 24))
    stride = int(ratios.get("stride", 24))
    warmup_days = int(ratios.get("warmup_days", 60))
    warmup_hours = warmup_days * 24
    conf = float(ratios.get("confidence_level", 0.80))
    alpha = 1.0 - conf

    fit_history_days = config.get("forecasting", {}).get("fit_history_days", None)
    fit_window_hours = None
    if fit_history_days is not None:
        try:
            fit_history_days = int(fit_history_days)
            fit_window_hours = fit_history_days * 24
            _log(
                f"[{cc or ''}] Using rolling fit window: last {fit_history_days} days of history per step"
            )
        except Exception:
            fit_window_hours = None

    first_ts = idx[0]
    first_valid_ts = first_ts + pd.Timedelta(hours=warmup_hours)

    def run_segment(
        start_ts: pd.Timestamp, end_ts: pd.Timestamp, label: str
    ) -> pd.DataFrame:
        rows = []
        step_ts = max(start_ts, first_valid_ts)
        if step_ts > end_ts:
            _log(f"[{cc or ''}][{label}] No steps (not enough warmup or empty segment)")
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "y_true",
                    "yhat",
                    "lo",
                    "hi",
                    "horizon",
                    "train_end",
                ]
            )

        remaining_hours = (end_ts - step_ts).total_seconds() / 3600.0
        total_steps = int(remaining_hours // stride) + 1
        progress_every = max(1, total_steps // 10)
        start_time = time.time()
        step_idx = 0
        avg_sec = None
        _log(
            f"[{cc or ''}][{label}] Starting backtest: {total_steps} steps from {step_ts} to {end_ts}"
        )
        while step_ts <= end_ts:
            loop_start = time.time()
            if fit_window_hours is not None:
                start_hist = max(idx[0], step_ts - pd.Timedelta(hours=fit_window_hours))
                y_hist = y.loc[start_hist:step_ts]
                ref_df = df.loc[start_hist:step_ts]
            else:
                y_hist = y.loc[:step_ts]
                ref_df = df.loc[:step_ts]
            X_hist = None
            Xf = None
            if config["forecasting"].get("use_exogenous", True):
                try:
                    from .exogenous_features import build_exogenous
                except ImportError:
                    from exogenous_features import build_exogenous
                X_hist = build_exogenous(
                    ref_df,
                    include_calendar=include_calendar,
                    include_wind=include_wind,
                    include_solar=include_solar,
                )
                Xf = build_future_exogenous(
                    step_ts,
                    periods=horizon,
                    include_calendar=include_calendar,
                    include_wind=include_wind,
                    include_solar=include_solar,
                    reference_df=ref_df,
                )
            try:
                model = SARIMAX(
                    y_hist,
                    exog=X_hist,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False)

                fc = res.get_forecast(steps=horizon, exog=Xf)
                mean = fc.predicted_mean
                conf_int = fc.conf_int(alpha=alpha)
                lo = conf_int.iloc[:, 0]
                hi = conf_int.iloc[:, 1]

                step_df = pd.DataFrame(
                    {
                        "timestamp": mean.index,
                        "yhat": mean.values,
                        "lo": lo.values,
                        "hi": hi.values,
                    }
                )
                step_df["horizon"] = np.arange(1, len(step_df) + 1)
                step_df["train_end"] = step_ts
                step_df = step_df.set_index("timestamp").join(y.rename("y_true"))
                step_df = step_df.reset_index()

                step_df = step_df[
                    (step_df["timestamp"] > step_ts) & (step_df["timestamp"] <= end_ts)
                ]
                rows.append(step_df)
            except Exception as e:
                _log(f"[{cc or ''}][{label}] step at {step_ts} failed: {e}")

            step_idx += 1
            elapsed = time.time() - loop_start
            if avg_sec is None:
                avg_sec = elapsed
            else:
                avg_sec = 0.9 * avg_sec + 0.1 * elapsed
            if (step_idx % progress_every == 0) or (step_idx == total_steps):
                eta_sec = max(0.0, (total_steps - step_idx) * (avg_sec or 0.0))
                _log(
                    f"[{cc or ''}][{label}] {step_idx}/{total_steps} steps | avg {avg_sec:.2f}s/step | ETA {eta_sec/60:.1f}m"
                )

            step_ts = step_ts + pd.Timedelta(hours=stride)

        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.sort_values(["timestamp", "horizon"], inplace=True)
            return out
        else:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "y_true",
                    "yhat",
                    "lo",
                    "hi",
                    "horizon",
                    "train_end",
                ]
            )

    dev_fc = run_segment(train_end_ts, dev_end_ts, label="DEV")
    test_fc = run_segment(dev_end_ts, idx[-1], label="TEST")
    return dev_fc, test_fc


def calculate_metrics(df: pd.DataFrame, seasonality: int = 24) -> Dict[str, float]:
    # Metrics: MASE(primary), sMAPE, MSE, RMSE, MAPE, 80% PI coverage
    y_true = df["y_true"].values
    yhat = df["yhat"].values
    lo = df["lo"].values
    hi = df["hi"].values

    # Drop NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(yhat)
    y_true = y_true[mask]
    yhat = yhat[mask]
    lo = lo[mask]
    hi = hi[mask]

    n = len(y_true)
    if n == 0:
        return {
            m: np.nan
            for m in ["MASE", "sMAPE", "MSE", "RMSE", "MAPE", "PI_80_Coverage"]
        }

    # MASE: mean absolute scaled error
    # scale by in-sample naive seasonal forecast error
    y = y_true  # current window, but normally use training series or stable base
    naive_forecast_error = (
        np.mean(np.abs(y[seasonality:] - y[:-seasonality]))
        if n > seasonality
        else 1e-10
    )
    mase = np.mean(np.abs(y_true - yhat)) / (naive_forecast_error + 1e-10)

    # sMAPE
    denom = (np.abs(y_true) + np.abs(yhat)) / 2
    smape = np.mean(np.abs(y_true - yhat) / (denom + 1e-10)) * 100

    # MSE/RMSE
    mse = np.mean((y_true - yhat) ** 2)
    rmse = np.sqrt(mse)

    # MAPE
    mape = np.mean(np.abs((y_true - yhat) / (y_true + 1e-10))) * 100

    # 80% PI coverage
    pi_covered = np.mean((y_true >= lo) & (y_true <= hi))

    return {
        "MASE": mase,
        "sMAPE": smape,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "PI_80_Coverage": pi_covered,
    }


def run(config_path: str = DEFAULT_CONFIG_PATH):
    _log("Reading config and preparing outputs...")
    _log(f"Config path: {config_path}")

    if not os.path.exists(config_path):
        _log(f"ERROR: Config file not found at {config_path}")
        # Try alternative path
        alt_config = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
        )
        if os.path.exists(alt_config):
            config_path = alt_config
            _log(f"Using alternative config path: {config_path}")
        else:
            _log("Cannot find config.yaml, exiting")
            return

    cfg = _read_config(config_path)
    outputs = cfg.get("outputs", {})
    out_folder = outputs.get("forecasts_folder") or outputs.get("folder", "outputs")

    # Create outputs folder in project root
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out_folder = os.path.join(root_dir, out_folder)
    os.makedirs(out_folder, exist_ok=True)
    _log(f"Output folder: {out_folder}")

    # Load cleaned data from data folder
    _log("Loading cleaned country data from data folder...")
    # Handle both possible config structures
    data_config = cfg.get("data", {})
    if isinstance(data_config, dict):
        data_folder = data_config.get("folder", "data")
    else:
        # If data is a string path, use it directly
        data_folder = data_config if isinstance(data_config, str) else "data"
    
    dfs_by_cc = load_cleaned_country_files(data_folder)

    if not dfs_by_cc:
        _log("ERROR: No country data loaded, exiting")
        return

    _log(f"Loaded countries: {list(dfs_by_cc.keys())}")

    fcfg = cfg.get("forecasting", {})
    use_exog = bool(fcfg.get("use_exogenous", True))
    include_calendar = (
        bool(fcfg.get("include_calendar_features", True)) if use_exog else False
    )
    include_wind = bool(fcfg.get("include_wind", False)) if use_exog else False
    include_solar = bool(fcfg.get("include_solar", False)) if use_exog else False
    _log(
        f"Exogenous: use={use_exog}, calendar={include_calendar}, wind={include_wind}, solar={include_solar}"
    )

    sarima_cfg = cfg.get("sarima", {})

    for cc, df in dfs_by_cc.items():
        _log(f"[{cc}] Preparing order selection and backtest...")
        df_idx = df.set_index("timestamp").sort_index()
        y_all = df_idx["load"].astype(float)
        n = len(y_all)
        n_train = int(n * float(fcfg.get("train_ratio", 0.8)))
        y_train = y_all.iloc[:n_train]
        X_train = None
        if use_exog:
            try:
                from .exogenous_features import build_exogenous
            except ImportError:
                from exogenous_features import build_exogenous
            X_train = build_exogenous(
                df_idx.iloc[:n_train], include_calendar, include_wind, include_solar
            )

        order, seasonal_order = select_sarima_order(y_train, X_train, sarima_cfg)
        _log(f"[{cc}] Selected order={order}, seasonal_order={seasonal_order}")

        dev_fc, test_fc = expanding_backtest(
            df,
            order,
            seasonal_order,
            cfg,
            include_calendar,
            include_wind,
            include_solar,
            cc=cc,
        )

        # Save forecasts
        dev_path = os.path.join(out_folder, f"{cc}_forecasts_dev.csv")
        test_path = os.path.join(out_folder, f"{cc}_forecasts_test.csv")
        dev_fc.to_csv(dev_path, index=False)
        test_fc.to_csv(test_path, index=False)
        _log(f"[{cc}] Wrote dev forecasts -> {dev_path} ({len(dev_fc)} rows)")
        _log(f"[{cc}] Wrote test forecasts -> {test_path} ({len(test_fc)} rows)")

        # Compute metrics for dev and test
        dev_metrics = calculate_metrics(dev_fc)
        test_metrics = calculate_metrics(test_fc)

        _log(f"[{cc}] DEV metrics: {dev_metrics}")
        _log(f"[{cc}] TEST metrics: {test_metrics}")


if __name__ == "__main__":
    run()
