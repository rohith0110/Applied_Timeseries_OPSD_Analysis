import pandas as pd
import numpy as np
from typing import Optional, Tuple


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=index)
    df['hour'] = index.hour
    df['dow'] = index.dayofweek
    # one-hots with full rank; model has intercept so drop_first=True
    hour_oh = pd.get_dummies(df['hour'], prefix='hr', drop_first=True)
    dow_oh = pd.get_dummies(df['dow'], prefix='dow', drop_first=True)
    out = pd.concat([hour_oh, dow_oh], axis=1)
    out.index = index
    return out


def build_exogenous(
    df: pd.DataFrame,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
) -> pd.DataFrame:
    """Create exogenous regressors aligned to df.index (timestamp).

    df: requires index=timestamp; optional columns 'wind','solar'.
    Returns empty DataFrame if no features selected.
    """
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    parts = []
    if include_calendar:
        parts.append(_calendar_features(df.index))
    if include_wind and 'wind' in df.columns:
        parts.append(df[['wind']].copy())
    if include_solar and 'solar' in df.columns:
        parts.append(df[['solar']].copy())
    if not parts:
        return pd.DataFrame(index=df.index)
    X = pd.concat(parts, axis=1)
    # fill any gaps
    return X.astype(float).fillna(method='ffill').fillna(0.0)


def build_future_exogenous(
    last_ts: pd.Timestamp,
    periods: int,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create exog for the next `periods` hours after last_ts.

    For wind/solar, if requested and available in reference_df, we use a simple
    hold-last strategy (repeat last observed value). If not available, zeros.
    """
    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=periods, freq='H')
    parts = []
    if include_calendar:
        parts.append(_calendar_features(future_index))

    if include_wind:
        if reference_df is not None and 'wind' in reference_df.columns and not reference_df['wind'].dropna().empty:
            last_wind = reference_df['wind'].dropna().iloc[-1]
        else:
            last_wind = 0.0
        parts.append(pd.DataFrame({'wind': np.repeat(last_wind, periods)}, index=future_index))

    if include_solar:
        if reference_df is not None and 'solar' in reference_df.columns and not reference_df['solar'].dropna().empty:
            last_solar = reference_df['solar'].dropna().iloc[-1]
        else:
            last_solar = 0.0
        parts.append(pd.DataFrame({'solar': np.repeat(last_solar, periods)}, index=future_index))

    if not parts:
        return pd.DataFrame(index=future_index)
    Xf = pd.concat(parts, axis=1)
    return Xf.astype(float)


def prepare_endog_exog(
    df: pd.DataFrame,
    include_calendar: bool = True,
    include_wind: bool = False,
    include_solar: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return (y, X) where y is load and X exogenous aligned to index.
    """
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    y = df['load'].astype(float)
    X = build_exogenous(df, include_calendar, include_wind, include_solar)
    # align
    X = X.reindex(y.index)
    return y, X
