"""
Trend Analysis
==============
Compute slopes, velocities, and accelerations from time-series data.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional
from scipy import stats


def compute_slope(
    df: pd.DataFrame,
    col: str,
    window: int,
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Compute the linear regression slope over a rolling window.
    
    This is the "trend direction" feature - positive means increasing,
    negative means decreasing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dense time-series DataFrame.
    col : str
        Column to compute slope for.
    window : int
        Rolling window size.
    entity_col : str or list of str, optional
        Entity column(s) for grouping.
    time_col : str
        Time column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new column: {col}_slope_{window}
        
    Example
    -------
    >>> df = compute_slope(df, col='uplink_bytes', window=7, entity_col='imsi')
    # Creates uplink_bytes_slope_7 showing 7-period trend direction
    """
    df = df.copy()
    
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    new_col_name = f"{col}_slope_{window}"
    
    def calc_slope(x):
        if len(x) < 2:
            return np.nan
        x_vals = np.arange(len(x))
        slope, _, _, _, _ = stats.linregress(x_vals, x)
        return slope
    
    sort_cols = (entity_col or []) + [time_col]
    df = df.sort_values(sort_cols)
    
    if entity_col:
        df[new_col_name] = df.groupby(entity_col)[col].rolling(
            window=window, min_periods=2
        ).apply(calc_slope, raw=True).reset_index(level=list(range(len(entity_col))), drop=True)
    else:
        df[new_col_name] = df[col].rolling(
            window=window, min_periods=2
        ).apply(calc_slope, raw=True)
    
    return df


def compute_velocity(
    df: pd.DataFrame,
    col: str,
    window: int = 1,
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Compute velocity (rate of change) for a column.
    
    Velocity = current_value - mean(previous window)
    
    Parameters
    ----------
    df : pd.DataFrame
        Dense time-series DataFrame.
    col : str
        Column to compute velocity for.
    window : int
        Window size for the baseline mean.
    entity_col : str or list of str, optional
        Entity column(s).
    time_col : str
        Time column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new column: {col}_velocity_{window}
        
    Example
    -------
    >>> df = compute_velocity(df, 'uplink_bytes', window=7, entity_col='imsi')
    """
    df = df.copy()
    
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    new_col_name = f"{col}_velocity_{window}"
    baseline_col = f"_baseline_{col}"
    
    sort_cols = (entity_col or []) + [time_col]
    df = df.sort_values(sort_cols)
    
    # Compute rolling mean of previous 'window' periods (shifted by 1)
    if entity_col:
        df[baseline_col] = df.groupby(entity_col)[col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
    else:
        df[baseline_col] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
    
    # Velocity = current - baseline
    df[new_col_name] = df[col] - df[baseline_col]
    
    # Clean up temp column
    df = df.drop(columns=[baseline_col])
    
    return df


def compute_acceleration(
    df: pd.DataFrame,
    col: str,
    window: int = 1,
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Compute acceleration (second-order derivative / change in velocity).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with velocity already computed (or will compute).
    col : str
        Original column (acceleration is d(velocity)/dt).
    window : int
        Window for velocity calculation.
    entity_col : str or list of str, optional
        Entity column(s).
    time_col : str
        Time column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new column: {col}_acceleration_{window}
    """
    # First compute velocity if not present
    velocity_col = f"{col}_velocity_{window}"
    if velocity_col not in df.columns:
        df = compute_velocity(df, col, window, entity_col, time_col)
    
    # Acceleration = diff of velocity
    accel_col = f"{col}_acceleration_{window}"
    
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    if entity_col:
        df[accel_col] = df.groupby(entity_col)[velocity_col].diff()
    else:
        df[accel_col] = df[velocity_col].diff()
    
    return df
