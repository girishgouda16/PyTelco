"""
Data Cleaning Utilities
=======================
Handle missing values, outliers, and data quality issues.
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict


def fill_missing(
    df: pd.DataFrame,
    cols: Optional[Union[str, List[str]]] = None,
    strategy: str = 'zero',
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: str = 'timestamp',
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fill missing values with various strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : str or list of str, optional
        Columns to fill. If None, fills all numeric columns.
    strategy : str
        Fill strategy:
        - 'zero': Fill with 0
        - 'mean': Fill with column mean (global or per-entity)
        - 'median': Fill with column median
        - 'forward': Forward fill (last valid value)
        - 'backward': Backward fill (next valid value)
        - 'interpolate': Linear interpolation
    entity_col : str or list of str, optional
        Entity column(s) for group-wise filling.
    time_col : str
        Time column (for sorting before forward/backward fill).
    limit : int, optional
        Maximum consecutive NaNs to fill.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with filled values.
        
    Example
    -------
    >>> df = fill_missing(df, cols=['uplink_bytes'], strategy='forward', entity_col='imsi')
    """
    df = df.copy()
    
    # Normalize inputs
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]
    
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    # Sort for time-based strategies
    if strategy in ['forward', 'backward', 'interpolate']:
        sort_cols = (entity_col or []) + [time_col]
        df = df.sort_values(sort_cols)
    
    for col in cols:
        if col not in df.columns:
            continue
            
        if strategy == 'zero':
            df[col] = df[col].fillna(0)
            
        elif strategy == 'mean':
            if entity_col:
                df[col] = df.groupby(entity_col)[col].transform(
                    lambda x: x.fillna(x.mean())
                )
            else:
                df[col] = df[col].fillna(df[col].mean())
                
        elif strategy == 'median':
            if entity_col:
                df[col] = df.groupby(entity_col)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            else:
                df[col] = df[col].fillna(df[col].median())
                
        elif strategy == 'forward':
            if entity_col:
                df[col] = df.groupby(entity_col)[col].ffill(limit=limit)
            else:
                df[col] = df[col].ffill(limit=limit)
                
        elif strategy == 'backward':
            if entity_col:
                df[col] = df.groupby(entity_col)[col].bfill(limit=limit)
            else:
                df[col] = df[col].bfill(limit=limit)
                
        elif strategy == 'interpolate':
            if entity_col:
                df[col] = df.groupby(entity_col)[col].transform(
                    lambda x: x.interpolate(method='linear', limit=limit)
                )
            else:
                df[col] = df[col].interpolate(method='linear', limit=limit)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def clip_outliers(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    method: str = 'percentile',
    lower: float = 0.01,
    upper: float = 0.99,
    entity_col: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Clip outliers in specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : str or list of str
        Columns to clip.
    method : str
        Clipping method:
        - 'percentile': Clip at lower/upper percentiles
        - 'iqr': Clip at Q1 - 1.5*IQR and Q3 + 1.5*IQR
        - 'zscore': Clip values beyond Z standard deviations
    lower : float
        Lower bound (percentile 0-1, or Z-score multiplier).
    upper : float
        Upper bound (percentile 0-1, or Z-score multiplier).
    entity_col : str or list of str, optional
        Compute bounds per entity (useful for heterogeneous populations).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with clipped values.
        
    Example
    -------
    >>> df = clip_outliers(df, cols=['uplink_bytes'], method='percentile', upper=0.99)
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    for col in cols:
        if col not in df.columns:
            continue
        
        if method == 'percentile':
            if entity_col:
                def clip_group(x):
                    q_low = x.quantile(lower)
                    q_high = x.quantile(upper)
                    return x.clip(q_low, q_high)
                df[col] = df.groupby(entity_col)[col].transform(clip_group)
            else:
                q_low = df[col].quantile(lower)
                q_high = df[col].quantile(upper)
                df[col] = df[col].clip(q_low, q_high)
                
        elif method == 'iqr':
            if entity_col:
                def clip_iqr(x):
                    q1, q3 = x.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    return x.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                df[col] = df.groupby(entity_col)[col].transform(clip_iqr)
            else:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                
        elif method == 'zscore':
            if entity_col:
                def clip_zscore(x):
                    mean, std = x.mean(), x.std()
                    return x.clip(mean - lower * std, mean + upper * std)
                df[col] = df.groupby(entity_col)[col].transform(clip_zscore)
            else:
                mean, std = df[col].mean(), df[col].std()
                df[col] = df[col].clip(mean - lower * std, mean + upper * std)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return df


def remove_inactive(
    df: pd.DataFrame,
    entity_col: Union[str, List[str]],
    value_col: str,
    threshold: float = 0.0,
    min_active_ratio: float = 0.1,
) -> pd.DataFrame:
    """
    Remove entities that are mostly inactive (useful for churn analysis).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dense time-series DataFrame.
    entity_col : str or list of str
        Entity identifier column(s).
    value_col : str
        Value column to check activity.
    threshold : float
        Values below this are considered "inactive".
    min_active_ratio : float
        Minimum ratio of active periods required (0-1).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with inactive entities removed.
        
    Example
    -------
    >>> df = remove_inactive(df, 'imsi', 'uplink_bytes', min_active_ratio=0.1)
    # Removes subscribers who were active less than 10% of the time
    """
    df = df.copy()
    
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    # Calculate active ratio per entity
    active = df.groupby(entity_col)[value_col].apply(
        lambda x: (x > threshold).mean()
    ).reset_index(name='_active_ratio')
    
    # Filter entities
    active_entities = active[active['_active_ratio'] >= min_active_ratio]
    
    # Merge back
    result = df.merge(active_entities[entity_col], on=entity_col, how='inner')
    
    return result
