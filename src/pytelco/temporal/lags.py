"""
Lag Feature Engineering
=======================
Add lagged, rolling, and differenced features to time-series data.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Callable


def add_lags(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    lags: List[int],
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
    prefix: str = 'lag'
) -> pd.DataFrame:
    """
    Add lagged versions of columns to the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input time-series DataFrame (should be dense/sorted).
    cols : str or list of str
        Column(s) to create lags for.
    lags : list of int
        Lag periods (e.g., [1, 7, 30] for t-1, t-7, t-30).
    entity_col : str or list of str, optional
        Entity column(s) to group by (e.g., 'imsi'). 
        If None, treats entire DataFrame as one entity.
    time_col : str
        Time column name (used for sorting).
    prefix : str
        Prefix for new column names.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional lag columns.
        
    Example
    -------
    >>> df = add_lags(df, cols=['uplink_bytes'], lags=[1, 24, 168], entity_col='imsi')
    # Creates: uplink_bytes_lag_1, uplink_bytes_lag_24, uplink_bytes_lag_168
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    # Sort by entity and time
    sort_cols = (entity_col or []) + [time_col]
    df = df.sort_values(sort_cols)
    
    for col in cols:
        for lag in lags:
            new_col_name = f"{col}_{prefix}_{lag}"
            if entity_col:
                df[new_col_name] = df.groupby(entity_col)[col].shift(lag)
            else:
                df[new_col_name] = df[col].shift(lag)
    
    return df


def add_rolling(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    windows: List[int],
    funcs: List[str] = ['mean', 'std'],
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Add rolling window statistics to the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input time-series DataFrame.
    cols : str or list of str
        Column(s) to compute rolling stats for.
    windows : list of int
        Window sizes (e.g., [7, 30] for 7-period and 30-period windows).
    funcs : list of str
        Aggregation functions: 'mean', 'std', 'sum', 'min', 'max', 'count'.
    entity_col : str or list of str, optional
        Entity column(s) to group by.
    time_col : str
        Time column name.
    min_periods : int
        Minimum observations required for a valid result.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional rolling columns.
        
    Example
    -------
    >>> df = add_rolling(df, cols=['uplink_bytes'], windows=[7, 30], funcs=['mean', 'std'])
    # Creates: uplink_bytes_rolling_7_mean, uplink_bytes_rolling_7_std, ...
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    # Sort
    sort_cols = (entity_col or []) + [time_col]
    df = df.sort_values(sort_cols)
    
    for col in cols:
        for window in windows:
            for func in funcs:
                new_col_name = f"{col}_rolling_{window}_{func}"
                
                if entity_col:
                    grouped = df.groupby(entity_col)[col]
                    rolling = grouped.rolling(window=window, min_periods=min_periods)
                else:
                    rolling = df[col].rolling(window=window, min_periods=min_periods)
                
                # Apply the aggregation
                if func == 'mean':
                    result = rolling.mean()
                elif func == 'std':
                    result = rolling.std()
                elif func == 'sum':
                    result = rolling.sum()
                elif func == 'min':
                    result = rolling.min()
                elif func == 'max':
                    result = rolling.max()
                elif func == 'count':
                    result = rolling.count()
                else:
                    raise ValueError(f"Unknown function: {func}")
                
                # Handle grouped rolling (returns MultiIndex)
                if entity_col:
                    df[new_col_name] = result.reset_index(level=list(range(len(entity_col))), drop=True)
                else:
                    df[new_col_name] = result
    
    return df


def add_diff(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    periods: List[int] = [1],
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Add differenced (delta) features to the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input time-series DataFrame.
    cols : str or list of str
        Column(s) to difference.
    periods : list of int
        Difference periods (e.g., [1] for first-order, [1, 2] for first and second).
    entity_col : str or list of str, optional
        Entity column(s) to group by.
    time_col : str
        Time column name.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with difference columns.
        
    Example
    -------
    >>> df = add_diff(df, cols=['uplink_bytes'], periods=[1], entity_col='imsi')
    # Creates: uplink_bytes_diff_1 (change from previous period)
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    sort_cols = (entity_col or []) + [time_col]
    df = df.sort_values(sort_cols)
    
    for col in cols:
        for period in periods:
            new_col_name = f"{col}_diff_{period}"
            if entity_col:
                df[new_col_name] = df.groupby(entity_col)[col].diff(periods=period)
            else:
                df[new_col_name] = df[col].diff(periods=period)
    
    return df
