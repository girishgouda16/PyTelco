"""
Data Loaders
============
Load various telco data formats into standardized DataFrames.
"""

import pandas as pd
import json
import glob
import os
from typing import Optional


def load_sip(
    path: str,
    format: str = 'auto',
) -> pd.DataFrame:
    """
    Load SIP signaling data from files.
    
    Parameters
    ----------
    path : str
        Path to file or directory (supports glob patterns).
    format : str
        File format: 'auto', 'jsonl', 'csv', 'parquet'.
        
    Returns
    -------
    pd.DataFrame
        Standardized SIP DataFrame.
    """
    return _load_data(path, format, expected_cols=[
        'timestamp', 'call_id', 'method', 'status_code', 
        'user_agent', 'contact_uri'
    ])


def load_gtpu(
    path: str,
    format: str = 'auto',
) -> pd.DataFrame:
    """
    Load GTP-U traffic data from files.
    
    Parameters
    ----------
    path : str
        Path to file or directory.
    format : str
        File format.
        
    Returns
    -------
    pd.DataFrame
        Standardized GTP-U DataFrame.
    """
    return _load_data(path, format, expected_cols=[
        'timestamp', 'teid', 'payload_length', 
        'inner_src_ip', 'inner_dest_port'
    ])


def load_cdr(
    path: str,
    format: str = 'auto',
) -> pd.DataFrame:
    """
    Load CDR data from files.
    
    Parameters
    ----------
    path : str
        Path to file or directory.
    format : str
        File format.
        
    Returns
    -------
    pd.DataFrame
        Standardized CDR DataFrame.
    """
    return _load_data(path, format, expected_cols=[
        'timestamp', 'imsi', 'imei', 'uplink_bytes', 
        'downlink_bytes', 'duration_sec'
    ])


def _load_data(
    path: str,
    format: str,
    expected_cols: list,
) -> pd.DataFrame:
    """Internal loader that handles multiple formats."""
    
    # Expand glob patterns
    if '*' in path:
        files = glob.glob(path)
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, '*'))
    else:
        files = [path]
    
    if not files:
        raise FileNotFoundError(f"No files found at: {path}")
    
    dfs = []
    for file_path in files:
        # Auto-detect format
        if format == 'auto':
            if file_path.endswith('.jsonl'):
                fmt = 'jsonl'
            elif file_path.endswith('.csv'):
                fmt = 'csv'
            elif file_path.endswith('.parquet'):
                fmt = 'parquet'
            else:
                fmt = 'csv'  # Default
        else:
            fmt = format
        
        # Load based on format
        if fmt == 'jsonl':
            records = []
            with open(file_path, 'r') as f:
                for line in f:
                    records.append(json.loads(line.strip()))
            df = pd.DataFrame(records)
        elif fmt == 'csv':
            df = pd.read_csv(file_path)
        elif fmt == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unknown format: {fmt}")
        
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamp
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    return result
