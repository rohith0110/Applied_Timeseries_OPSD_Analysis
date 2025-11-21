"""Classical SARIMAX backtesting entry point."""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.statespace.sarimax import SARIMAX


try:
    from pmdarima import auto_arima

    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    auto_arima = None

try:
    from ..metrics.metrics import compute_all_metrics
except ImportError:
    from metrics.metrics import compute_all_metrics


warnings.filterwarnings("ignore", category=UserWarning)
DEFAULT_CONFIG_PATH = "config.yaml"


def _log(msg: str) -> None:
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


def _read_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    data = {
        "sin_hour": np.sin(2 * np.pi * hours / 24.0),
        "cos_hour": np.cos(2 * np.pi * hours / 24.0),
        "sin_dow": np.sin(2 * np.pi * dow / 7.0),
        "cos_dow": np.cos(2 * np.pi * dow / 7.0),
        "is_weekend": (dow >= 5).astype(float),
    }

    for d in range(7):
        data[f"dow_{d}"] = (dow == d).astype(float)
    return pd.DataFrame(data, index=index)


def build_exog_frame(
    df: pd.DataFrame,
    value_cols: Optional[List[str]],
    include_calendar: bool,
) -> Optional[pd.DataFrame]:
    parts: List[pd.DataFrame] = []
    if value_cols:
        available = [col for col in value_cols if col in df.columns]
        if available:
            parts.append(df[available].astype(float))
    if include_calendar:
        parts.append(_build_calendar_frame(df.index))
    if not parts:
        return None
    return pd.concat(parts, axis=1)


def build_future_exog(
    start_ts: pd.Timestamp,
    horizon: int,
    include_calendar: bool,
    value_cols: Optional[List[str]],
    reference_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    future_index = pd.date_range(
        start_ts + pd.Timedelta(hours=1), periods=horizon, freq="H"
    )
    parts: List[pd.DataFrame] = []
    if value_cols:
        available = [col for col in value_cols if col in reference_df.columns]
        if available:
            last_values = reference_df[available].iloc[-1].to_dict()
            values = {
                col: np.repeat(last_values[col], horizon).astype(float)
                for col in available
            }
            parts.append(pd.DataFrame(values, index=future_index))
    if include_calendar:
        parts.append(_build_calendar_frame(future_index))
    if not parts:
        return None
    return pd.concat(parts, axis=1)


def auto_arima_order(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    sarima_cfg: dict,
    max_p: int = 5,
    max_q: int = 5,
    max_P: int = 2,
    max_Q: int = 2,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Use pmdarima auto_arima for intelligent order selection.

    This is much faster than grid search and often finds better orders.
    Uses stepwise algorithm to efficiently explore the order space.
    """
    if not PMDARIMA_AVAILABLE:
        _log("⚠️  pmdarima not available, falling back to grid search")
        return select_sarima_order(y, exog, sarima_cfg)

    s = int(sarima_cfg.get("s", 24))
    _log(f"Auto-ARIMA order selection (stepwise, m={s})...")

    tic = time.time()
    try:
        model = auto_arima(
            y,
            exogenous=exog,
            seasonal=True,
            m=s,
            max_p=max_p,
            max_q=max_q,
            max_P=max_P,
            max_Q=max_Q,
            max_d=2,
            max_D=1,
            start_p=1,
            start_q=1,
            start_P=1,
            start_Q=1,
            stepwise=True,
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aicc",
            method="lbfgs",
            maxiter=50,
        )
        elapsed = time.time() - tic

        order = model.order
        seasonal_order = model.seasonal_order
        aicc = model.aicc()

        _log(
            f"Auto-ARIMA completed in {elapsed:.1f}s | order={order} seasonal={seasonal_order} AICc={aicc:.1f}"
        )
        return (order, seasonal_order)

    except Exception as e:
        _log(f"Auto-ARIMA failed ({e}), falling back to grid search")
        return select_sarima_order(y, exog, sarima_cfg)


def select_sarima_order(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    sarima_cfg: dict,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    """Grid-based SARIMAX order selection using AICc."""
    p_range = sarima_cfg.get("p_range", [0, 1, 2])
    d_range = sarima_cfg.get("d_range", [0, 1])
    q_range = sarima_cfg.get("q_range", [0, 1, 2])
    P_range = sarima_cfg.get("P_range", [0, 1])
    D_range = sarima_cfg.get("D_range", [0, 1])
    Q_range = sarima_cfg.get("Q_range", [0, 1])
    s = int(sarima_cfg.get("s", 24))

    best = None
    best_metrics = (np.inf, np.inf)
    total = (
        len(p_range)
        * len(d_range)
        * len(q_range)
        * len(P_range)
        * len(D_range)
        * len(Q_range)
    )
    checked = 0
    interval = max(1, total // 5)
    _log(f"Order search: {total} combinations")
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
                                    exog=exog,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                )
                                res = model.fit(disp=False)

                                k = res.params.shape[0]
                                n = len(y)
                                aic = res.aic
                                aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
                                if aicc < best_metrics[0]:
                                    best_metrics = (aicc, aic)
                                    best = (order, seasonal_order)
                            except Exception:
                                pass
                            checked += 1
                            if checked % interval == 0 or checked == total:
                                status = f"Order search progress: {checked}/{total}"
                                if best:
                                    status += f" | best order={best[0]} seasonal={best[1]} AICc={best_metrics[0]:.1f}"
                                _log(status)
    if best is None:
        best = ((1, 1, 0), (1, 1, 0, s))
    return best


def expanding_backtest(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    config: dict,
    exog_cols: Optional[List[str]],
    include_calendar: bool,
    cc: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if "timestamp" in df.columns:
        df.set_index("timestamp", inplace=True)
    df = df.asfreq("H").interpolate()

    y = np.log1p(df["load"].astype(float))

    ratios = config.get("forecasting", {})
    train_ratio = float(ratios.get("train_ratio", 0.8))
    val_ratio = float(ratios.get("val_ratio", 0.1))
    horizon = int(ratios.get("horizon", 24))
    stride = int(ratios.get("stride", 24))
    warmup_days = int(ratios.get("warmup_days", 60))
    warmup_hours = warmup_days * 24

    n = len(y)
    n_train = max(int(n * train_ratio), horizon)
    n_val = int(n * val_ratio)

    timestamps = y.index
    train_end_ts = timestamps[n_train - 1]
    dev_end_ts = timestamps[min(n_train + n_val - 1, n - 1)]

    first_valid_ts = timestamps[0] + pd.Timedelta(hours=warmup_hours)

    def run_segment(
        start_ts: pd.Timestamp, end_ts: pd.Timestamp, label: str
    ) -> pd.DataFrame:
        rows: List[pd.DataFrame] = []
        step_ts = max(start_ts, first_valid_ts)
        if step_ts > end_ts:
            _log(f"[{cc}][{label}] No steps (insufficient warmup)")
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

        total_steps = int(((end_ts - step_ts).total_seconds() // 3600) // stride) + 1
        progress_every = max(1, total_steps // 10)
        step_idx = 0
        avg_sec: Optional[float] = None
        _log(
            f"[{cc}][{label}] Backtest from {step_ts} to {end_ts} ({total_steps} steps)"
        )

        while step_ts <= end_ts:
            tic = time.time()
            y_hist = y.loc[:step_ts]
            ref_df = df.loc[:step_ts]

            X_hist = build_exog_frame(ref_df, exog_cols, include_calendar)
            X_future = build_future_exog(
                step_ts, horizon, include_calendar, exog_cols, ref_df
            )

            try:
                model = SARIMAX(
                    y_hist,
                    exog=X_hist,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                res = model.fit(disp=False)
                fc = res.get_forecast(steps=horizon, exog=X_future)
                mean = fc.predicted_mean
                conf_int = fc.conf_int(
                    alpha=1 - float(ratios.get("confidence_level", 0.8))
                )

                step_df = pd.DataFrame(
                    {
                        "timestamp": mean.index,
                        "yhat": np.expm1(mean.values),
                        "lo": np.expm1(conf_int.iloc[:, 0].values),
                        "hi": np.expm1(conf_int.iloc[:, 1].values),
                        "horizon": np.arange(1, horizon + 1),
                        "train_end": step_ts,
                    }
                )
                step_df = step_df.set_index("timestamp").join(y.rename("y_true"))
                step_df = step_df.reset_index()
                step_df = step_df[
                    (step_df["timestamp"] > step_ts) & (step_df["timestamp"] <= end_ts)
                ]
                rows.append(step_df)
            except Exception as exc:
                _log(f"[{cc}][{label}] step {step_ts} failed: {exc}")

            step_idx += 1
            elapsed = time.time() - tic
            avg_sec = elapsed if avg_sec is None else 0.9 * avg_sec + 0.1 * elapsed
            if (step_idx % progress_every == 0) or (step_idx == total_steps):
                eta = max(0.0, (total_steps - step_idx) * (avg_sec or 0.0))
                _log(
                    f"[{cc}][{label}] {step_idx}/{total_steps} steps | avg {avg_sec:.2f}s | ETA {eta/60:.1f}m"
                )

            step_ts = step_ts + pd.Timedelta(hours=stride)

        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.sort_values(["timestamp", "horizon"], inplace=True)
            return out
        return pd.DataFrame(
            columns=["timestamp", "y_true", "yhat", "lo", "hi", "horizon", "train_end"]
        )

    dev_fc = run_segment(train_end_ts, dev_end_ts, "DEV")
    test_fc = run_segment(dev_end_ts, timestamps[-1], "TEST")
    return dev_fc, test_fc


def load_country_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


def run(
    config_path: str, countries: Optional[List[str]] = None, use_auto_arima: bool = True
) -> None:
    _log("Loading config...")
    cfg = _read_config(config_path)

    cleaned_files = cfg.get("cleaned_files")
    if not cleaned_files:
        raise ValueError("config.yaml must provide cleaned_files list")

    target_paths = [Path(p) for p in cleaned_files]
    if countries:
        normalized = {c.lower() for c in countries}
        target_paths = [
            p for p in target_paths if p.stem.split("_")[0].lower() in normalized
        ]
        if not target_paths:
            raise ValueError("No matching cleaned files for requested countries")

    output_dir = Path(cfg.get("output_dir", "outputs"))
    forecasts_dir = output_dir / "forecasts"
    metrics_dir = output_dir / "metrics"
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    exog_cols = cfg.get("exog_columns", [])
    include_calendar = bool(
        cfg.get("forecasting", {}).get("include_calendar_features", True)
    )
    sarima_cfg = cfg.get("sarima", {})
    sarima_best = cfg.get("sarima_best")

    summary_rows = []

    for csv_path in target_paths:
        cc = csv_path.stem.split("_")[0].upper()
        _log(f"[{cc}] Loading data from {csv_path}")
        df = load_country_df(csv_path)

        if sarima_best:
            order = (sarima_best["p"], sarima_best["d"], sarima_best["q"])
            seasonal_order = (
                sarima_best["P"],
                sarima_best["D"],
                sarima_best["Q"],
                sarima_best["s"],
            )
            _log(f"[{cc}] Using provided order={order} seasonal={seasonal_order}")
        else:
            df_idx = df.set_index("timestamp").sort_index()
            train_frac = float(cfg.get("forecasting", {}).get("train_ratio", 0.8))
            y_train = np.log1p(df_idx["load"].iloc[: int(len(df_idx) * train_frac)])
            X_train = build_exog_frame(
                df_idx.iloc[: len(y_train)], exog_cols, include_calendar
            )
            if use_auto_arima:
                order, seasonal_order = auto_arima_order(y_train, X_train, sarima_cfg)
            else:
                order, seasonal_order = select_sarima_order(
                    y_train, X_train, sarima_cfg
                )
            _log(f"[{cc}] Selected order={order} seasonal={seasonal_order}")

        dev_fc, test_fc = expanding_backtest(
            df,
            order,
            seasonal_order,
            cfg,
            exog_cols,
            include_calendar,
            cc,
        )

        dev_path = forecasts_dir / f"{cc}_SARIMAX_dev.csv"
        test_path = forecasts_dir / f"{cc}_SARIMAX_test.csv"
        dev_fc.to_csv(dev_path, index=False)
        test_fc.to_csv(test_path, index=False)
        _log(f"[{cc}] Saved dev forecasts -> {dev_path} ({len(dev_fc)} rows)")
        _log(f"[{cc}] Saved test forecasts -> {test_path} ({len(test_fc)} rows)")

        dev_metrics = compute_all_metrics(
            dev_fc["y_true"].values,
            dev_fc["yhat"].values,
            lower=dev_fc.get("lo"),
            upper=dev_fc.get("hi"),
            m=24,
        )
        test_metrics = compute_all_metrics(
            test_fc["y_true"].values,
            test_fc["yhat"].values,
            lower=test_fc.get("lo"),
            upper=test_fc.get("hi"),
            m=24,
        )

        pd.DataFrame(
            [
                {"country": cc, "model": "SARIMAX", "split": "dev", **dev_metrics},
                {"country": cc, "model": "SARIMAX", "split": "test", **test_metrics},
            ]
        ).to_csv(metrics_dir / f"{cc}_SARIMAX_metrics.csv", index=False)

        summary_rows.append({"country": cc, "split": "dev", **dev_metrics})
        summary_rows.append({"country": cc, "split": "test", **test_metrics})
        _log(f"[{cc}] DEV metrics: {dev_metrics}")
        _log(f"[{cc}] TEST metrics: {test_metrics}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            metrics_dir / "SARIMAX_summary.csv", index=False
        )
        _log("Wrote SARIMAX_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical SARIMAX backtest")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--countries",
        nargs="*",
        help="Optional list of country codes (prefixes of cleaned CSV names)",
    )
    parser.add_argument(
        "--auto-arima",
        action="store_true",
        default=True,
        help="Use pmdarima auto_arima for intelligent order selection (default)",
    )
    parser.add_argument(
        "--no-auto-arima",
        dest="auto_arima",
        action="store_false",
        help="Disable auto_arima, use grid search instead",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config, countries=args.countries, use_auto_arima=args.auto_arima)
