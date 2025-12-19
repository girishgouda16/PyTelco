"""
CDR Feature Extraction
======================
Domain-specific metrics for Call Detail Records.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


def compute_cdr_metrics(
    df: pd.DataFrame,
    entity_col: str = 'imsi',
    time_col: str = 'timestamp',
    freq: str = '1D',
) -> pd.DataFrame:
    """
    Compute CDR-specific metrics from billing data.
    
    Parameters
    ----------
    df : pd.DataFrame
        CDR data with columns: timestamp, imsi, imei, uplink_bytes, 
        downlink_bytes, duration_sec, first_cell_id, last_cell_id.
    entity_col : str
        Entity to aggregate by (usually 'imsi').
    time_col : str
        Timestamp column.
    freq : str
        Aggregation frequency.
        
    Returns
    -------
    pd.DataFrame
        Aggregated CDR metrics per entity per time bucket.
        
    Metrics
    -------
    - total_bytes: UL + DL combined
    - total_duration_sec: Sum of session durations
    - session_count: Number of sessions
    - ul_dl_ratio: Uplink/Downlink balance
    - unique_cells: Mobility indicator
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col]).dt.floor(freq)
    
    grouper = df.groupby([time_col, entity_col])
    
    result = pd.DataFrame()
    
    # Volume metrics
    if 'uplink_bytes' in df.columns and 'downlink_bytes' in df.columns:
        result['uplink_bytes'] = grouper['uplink_bytes'].sum()
        result['downlink_bytes'] = grouper['downlink_bytes'].sum()
        result['total_bytes'] = result['uplink_bytes'] + result['downlink_bytes']
        result['ul_dl_ratio'] = result['uplink_bytes'] / result['downlink_bytes'].replace(0, 1)
    
    # Duration
    if 'duration_sec' in df.columns:
        result['total_duration_sec'] = grouper['duration_sec'].sum()
    
    # Session count
    result['session_count'] = grouper.size()
    
    # Mobility (unique cells)
    if 'first_cell_id' in df.columns:
        result['unique_start_cells'] = grouper['first_cell_id'].nunique()
    if 'last_cell_id' in df.columns:
        result['unique_end_cells'] = grouper['last_cell_id'].nunique()
    
    return result.reset_index()


def compute_device_metrics(
    df: pd.DataFrame,
    imsi_col: str = 'imsi',
    imei_col: str = 'imei',
) -> pd.DataFrame:
    """
    Extract device-level metrics from CDR data.
    
    Parameters
    ----------
    df : pd.DataFrame
        CDR data with imsi and imei columns.
    imsi_col : str
        IMSI column name.
    imei_col : str
        IMEI column name.
        
    Returns
    -------
    pd.DataFrame
        Device metrics per IMSI.
        
    Metrics
    -------
    - device_count: Number of unique devices used
    - primary_device: Most frequently used IMEI
    - tac: Type Allocation Code (first 8 digits of IMEI)
    """
    df = df.copy()
    
    result = df.groupby(imsi_col).agg({
        imei_col: ['nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None]
    }).reset_index()
    
    result.columns = [imsi_col, 'device_count', 'primary_device']
    
    # Extract TAC (first 8 digits)
    result['primary_tac'] = result['primary_device'].str[:8]
    
    return result


def compute_usage_profile(
    df: pd.DataFrame,
    entity_col: str = 'imsi',
    time_col: str = 'timestamp',
    value_col: str = 'total_bytes',
) -> pd.DataFrame:
    """
    Compute usage profile statistics per entity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Aggregated usage data (output of to_dense_timeseries + compute_cdr_metrics).
    entity_col : str
        Entity column.
    time_col : str
        Time column.
    value_col : str
        Value column to profile.
        
    Returns
    -------
    pd.DataFrame
        Per-entity profile with mean, std, min, max, etc.
    """
    result = df.groupby(entity_col)[value_col].agg([
        'mean', 'std', 'min', 'max', 'sum', 'count'
    ]).reset_index()
    
    result.columns = [entity_col, f'{value_col}_mean', f'{value_col}_std',
                     f'{value_col}_min', f'{value_col}_max', 
                     f'{value_col}_total', f'{value_col}_periods']
    
    # Coefficient of variation (normalized volatility)
    result[f'{value_col}_cv'] = result[f'{value_col}_std'] / result[f'{value_col}_mean'].replace(0, 1)
    
    return result
