"""
Time Series Preprocessing
=========================
Convert sparse telco events into dense, analysis-ready time-series.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional


def align_to_grid(
    df: pd.DataFrame,
    time_col: str = 'timestamp',
    freq: str = '1H'
) -> pd.DataFrame:
    """
    Align timestamps to a regular time grid.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a timestamp column.
    time_col : str
        Name of the timestamp column.
    freq : str
        Frequency string (e.g., '1H', '15min', '1D').
        
    Returns
    -------
    pd.DataFrame
        DataFrame with timestamps floored to the grid.
        
    Example
    -------
    >>> df = align_to_grid(raw_df, time_col='timestamp', freq='1H')
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df[time_col] = df[time_col].dt.floor(freq)
    return df


def to_dense_timeseries(
    df: pd.DataFrame,
    entity_cols: Union[str, List[str]],
    time_col: str = 'timestamp',
    value_cols: Optional[List[str]] = None,
    freq: str = '1H',
    agg_func: str = 'sum',
    fill_value: float = 0.0,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert sparse event data into a dense, regularly-spaced time-series.
    
    This is THE foundational function for telco time-series analysis.
    It handles:
    1. Aggregating events into time buckets
    2. Creating a complete time grid (no gaps)
    3. Zero-filling periods with no activity
    
    Parameters
    ----------
    df : pd.DataFrame
        Sparse event data (e.g., CDRs, packets).
    entity_cols : str or list of str
        Column(s) identifying the entity (e.g., ['imsi'], ['imsi', 'imei']).
    time_col : str
        Name of the timestamp column.
    value_cols : list of str, optional
        Columns to aggregate. If None, creates a 'count' column.
    freq : str
        Time frequency for the grid (e.g., '5min', '1H', '1D').
    agg_func : str
        Aggregation function ('sum', 'mean', 'count', 'max', 'min').
    fill_value : float
        Value to use for missing time periods (typically 0).
    start_time : str, optional
        Override the start of the time range (ISO format).
    end_time : str, optional
        Override the end of the time range (ISO format).
        
    Returns
    -------
    pd.DataFrame
        Dense time-series with columns: [entity_cols, time_col] + value_cols.
        Every entity has a row for every time step in the range.
        
    Example
    -------
    >>> dense_df = to_dense_timeseries(
    ...     cdr_df,
    ...     entity_cols=['imsi'],
    ...     time_col='timestamp',
    ...     value_cols=['uplink_bytes', 'downlink_bytes'],
    ...     freq='1H',
    ...     fill_value=0
    ... )
    """
    df = df.copy()
    
    # Normalize inputs
    if isinstance(entity_cols, str):
        entity_cols = [entity_cols]
    
    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Align to grid
    df[time_col] = df[time_col].dt.floor(freq)
    
    # Determine time range
    if start_time:
        t_min = pd.Timestamp(start_time).floor(freq)
    else:
        t_min = df[time_col].min()
        
    if end_time:
        t_max = pd.Timestamp(end_time).floor(freq)
    else:
        t_max = df[time_col].max()
    
    # Create complete time grid
    full_time_range = pd.date_range(start=t_min, end=t_max, freq=freq)
    
    # Get unique entities
    unique_entities = df[entity_cols].drop_duplicates()
    
    # Create cartesian product: all entities x all times
    full_index = pd.MultiIndex.from_product(
        [unique_entities[col].unique() for col in entity_cols] + [full_time_range],
        names=entity_cols + [time_col]
    )
    full_grid = pd.DataFrame(index=full_index).reset_index()
    
    # Aggregate the input data
    group_cols = entity_cols + [time_col]
    
    if value_cols is None:
        # Just count events
        agg_df = df.groupby(group_cols).size().reset_index(name='event_count')
        value_cols = ['event_count']
    else:
        agg_dict = {col: agg_func for col in value_cols}
        agg_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Merge with full grid (left join to keep all grid points)
    result = full_grid.merge(agg_df, on=group_cols, how='left')
    
    # Fill missing values
    for col in value_cols:
        result[col] = result[col].fillna(fill_value)
    
    # Sort by entity and time
    result = result.sort_values(group_cols).reset_index(drop=True)
    
    return result
