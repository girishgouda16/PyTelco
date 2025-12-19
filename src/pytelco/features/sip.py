"""
SIP Feature Extraction
======================
Domain-specific metrics for SIP signaling data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List


def compute_sip_metrics(
    df: pd.DataFrame,
    entity_col: str = 'contact_uri',
    time_col: str = 'timestamp',
    freq: str = '1H',
) -> pd.DataFrame:
    """
    Compute SIP-specific metrics from signaling data.
    
    Parameters
    ----------
    df : pd.DataFrame
        SIP event data with columns: timestamp, call_id, method, status_code, 
        user_agent, contact_uri.
    entity_col : str
        Entity to aggregate by (usually 'contact_uri' or source IP).
    time_col : str
        Timestamp column.
    freq : str
        Aggregation frequency.
        
    Returns
    -------
    pd.DataFrame
        Aggregated SIP metrics per entity per time bucket.
        
    Metrics
    -------
    - total_requests: Total SIP events
    - success_ratio: 2xx responses / total events
    - error_ratio: 4xx+5xx+6xx / total events
    - unique_sessions: Distinct call_ids
    - events_per_sec: Request rate
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col]).dt.floor(freq)
    
    # Fill missing status codes (for requests)
    if 'status_code' in df.columns:
        df['status_code'] = df['status_code'].fillna(0)
    
    grouper = df.groupby([time_col, entity_col])
    
    result = pd.DataFrame()
    result['total_requests'] = grouper.size()
    
    if 'status_code' in df.columns:
        result['success_count'] = grouper['status_code'].apply(
            lambda x: ((x >= 200) & (x < 300)).sum()
        )
        result['error_count'] = grouper['status_code'].apply(
            lambda x: (x >= 400).sum()
        )
        result['success_ratio'] = result['success_count'] / result['total_requests']
        result['error_ratio'] = result['error_count'] / result['total_requests']
    
    if 'call_id' in df.columns:
        result['unique_sessions'] = grouper['call_id'].nunique()
    
    window_seconds = pd.Timedelta(freq).total_seconds()
    result['events_per_sec'] = result['total_requests'] / window_seconds
    
    return result.reset_index()


def compute_inter_arrival_time(
    df: pd.DataFrame,
    entity_col: str = 'contact_uri',
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Compute inter-arrival time statistics per entity.
    
    Critical for detecting flooding attacks (very regular IAT)
    vs normal traffic (high variance IAT).
    
    Parameters
    ----------
    df : pd.DataFrame
        SIP event data.
    entity_col : str
        Entity column.
    time_col : str
        Timestamp column.
        
    Returns
    -------
    pd.DataFrame
        Per-entity IAT statistics: mean, std, min, max.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([entity_col, time_col])
    
    # Compute time diff within each entity
    df['iat_seconds'] = df.groupby(entity_col)[time_col].diff().dt.total_seconds()
    
    result = df.groupby(entity_col)['iat_seconds'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    result.columns = [entity_col, 'iat_mean_sec', 'iat_std_sec', 
                      'iat_min_sec', 'iat_max_sec', 'event_count']
    
    return result
