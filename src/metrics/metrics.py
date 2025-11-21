"""
Forecasting Metrics
-------------------
Standard metrics for time series forecast evaluation.
"""

import numpy as np
import pandas as pd


def mase(y_true, y_pred, m=24):
    """
    Mean Absolute Scaled Error

    Args:
        y_true: actual values
        y_pred: predicted values
        m: seasonality period (default 24 for hourly data with daily seasonality)

    Returns:
        MASE score (lower is better, < 1 means better than naive seasonal forecast)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))

    if len(y_true) <= m:

        naive_error = np.mean(np.abs(np.diff(y_true)))
    else:
        naive_error = np.mean(np.abs(y_true[m:] - y_true[:-m]))

    if naive_error == 0:
        return np.inf if mae > 0 else 0.0

    return mae / naive_error


def smape(y_true, y_pred, epsilon=1e-8):
    """
    Stabilized Symmetric Mean Absolute Percentage Error

    Uses bounded denominator to avoid instability on small load values.
    This significantly improves sMAPE stability for electricity load data.

    Args:
        y_true: actual values
        y_pred: predicted values
        epsilon: minimum denominator value for numerical stability

    Returns:
        sMAPE as percentage (0-100, lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    denominator = np.maximum(denominator, epsilon)

    return 100 * np.mean(numerator / denominator)


def mse(y_true, y_pred):
    """Mean Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error

    Returns:
        MAPE as percentage (0-100, lower is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0
    if not mask.any():
        return 0.0

    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def pi_coverage(y_true, lower, upper):
    """
    Prediction Interval Coverage

    Computes the proportion of actual values that fall within the prediction interval.

    Args:
        y_true: actual values
        lower: lower bound of prediction interval
        upper: upper bound of prediction interval

    Returns:
        Coverage ratio (0-1, higher is better for confidence intervals)
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))

    if not mask.any():
        return np.nan

    y_true = y_true[mask]
    lower = lower[mask]
    upper = upper[mask]

    within_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(within_interval)


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def compute_all_metrics(y_true, y_pred, lower=None, upper=None, m=24):
    """
    Compute all standard forecasting metrics

    Args:
        y_true: actual values
        y_pred: predicted values
        lower: optional lower bound of prediction interval
        upper: optional upper bound of prediction interval
        m: seasonality period for MASE

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        "MASE": mase(y_true, y_pred, m=m),
        "sMAPE": smape(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }

    if lower is not None and upper is not None:
        metrics["80_PI_coverage"] = pi_coverage(y_true, lower, upper)
    else:
        metrics["80_PI_coverage"] = np.nan

    return metrics
