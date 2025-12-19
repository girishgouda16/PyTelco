"""
GTP-U Feature Extraction
========================
Domain-specific metrics for user-plane traffic data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List
from scipy.stats import entropy


def compute_gtpu_metrics(
    df: pd.DataFrame,
    entity_col: str = 'teid',
    time_col: str = 'timestamp',
    freq: str = '1H',
) -> pd.DataFrame:
    """
    Compute GTP-U specific metrics from traffic data.
    
    Parameters
    ----------
    df : pd.DataFrame
        GTP-U data with columns: timestamp, teid, payload_length, 
        inner_src_ip, inner_dest_port.
    entity_col : str
        Entity to aggregate by (usually 'teid' or 'inner_src_ip').
    time_col : str
        Timestamp column.
    freq : str
        Aggregation frequency.
        
    Returns
    -------
    pd.DataFrame
        Aggregated traffic metrics per entity per time bucket.
        
    Metrics
    -------
    - total_bytes: Sum of payload_length
    - throughput_bps: Bits per second
    - packet_count: Number of packets
    - payload_entropy: Shannon entropy of packet sizes (M2M detector)
    - small_packet_ratio: % packets < 64 bytes
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col]).dt.floor(freq)
    
    grouper = df.groupby([time_col, entity_col])
    window_seconds = pd.Timedelta(freq).total_seconds()
    
    result = pd.DataFrame()
    result['total_bytes'] = grouper['payload_length'].sum()
    result['throughput_bps'] = (result['total_bytes'] * 8) / window_seconds
    result['packet_count'] = grouper.size()
    result['packet_size_mean'] = grouper['payload_length'].mean()
    result['packet_size_std'] = grouper['payload_length'].std()
    
    # Payload entropy (key M2M vs Human classifier)
    def calc_entropy(x):
        counts = x.value_counts()
        return entropy(counts) if len(counts) > 1 else 0.0
    
    result['payload_entropy'] = grouper['payload_length'].apply(calc_entropy)
    
    # Small packet ratio (heartbeat/keepalive indicator)
    result['small_packet_ratio'] = grouper['payload_length'].apply(
        lambda x: (x < 64).mean()
    )
    
    if 'inner_dest_port' in df.columns:
        result['dominant_port'] = grouper['inner_dest_port'].apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        )
    
    return result.reset_index()


def compute_packet_jitter(
    df: pd.DataFrame,
    entity_col: str = 'teid',
    time_col: str = 'timestamp',
) -> pd.DataFrame:
    """
    Compute packet arrival jitter (variance in inter-packet gap).
    
    Critical for M2M classification:
    - Low jitter (regular intervals) = Machine
    - High jitter (bursty) = Human
    
    Parameters
    ----------
    df : pd.DataFrame
        Packet-level data.
    entity_col : str
        Session identifier.
    time_col : str
        Timestamp column.
        
    Returns
    -------
    pd.DataFrame
        Per-entity jitter statistics.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([entity_col, time_col])
    
    # Compute inter-packet gap
    df['ipg_ms'] = df.groupby(entity_col)[time_col].diff().dt.total_seconds() * 1000
    
    result = df.groupby(entity_col)['ipg_ms'].agg([
        'mean', 'std', 'var'
    ]).reset_index()
    
    result.columns = [entity_col, 'ipg_mean_ms', 'ipg_std_ms', 'ipg_var_ms']
    
    # Jitter is typically the variance or std of the IPG
    result['jitter_ms'] = result['ipg_std_ms']
    
    return result
