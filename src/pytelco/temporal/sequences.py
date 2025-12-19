"""
Sequence Extraction
===================
Create sequences for deep learning models (LSTM, Transformer).
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional, Tuple


def extract_sequences(
    df: pd.DataFrame,
    entity_col: Union[str, List[str]],
    feature_cols: List[str],
    seq_length: int,
    time_col: str = 'timestamp',
    target_col: Optional[str] = None,
    step: int = 1,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract fixed-length sequences from time-series data for deep learning.
    
    This creates sliding window sequences suitable for LSTM, GRU, or 
    Transformer models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dense time-series DataFrame (use to_dense_timeseries first).
    entity_col : str or list of str
        Entity identifier column(s) (e.g., 'imsi').
    feature_cols : list of str
        Columns to include as features in the sequence.
    seq_length : int
        Number of time steps in each sequence.
    time_col : str
        Time column name (for sorting).
    target_col : str, optional
        If provided, extracts the target value at sequence end.
    step : int
        Stride between sequences (1 = overlapping, seq_length = non-overlapping).
        
    Returns
    -------
    np.ndarray or tuple of np.ndarray
        If target_col is None:
            X with shape (n_sequences, seq_length, n_features)
        If target_col is provided:
            (X, y) where X has shape above and y has shape (n_sequences,)
            
    Example
    -------
    >>> X, y = extract_sequences(
    ...     dense_df,
    ...     entity_col='imsi',
    ...     feature_cols=['uplink_bytes', 'downlink_bytes'],
    ...     seq_length=48,  # 48 hours of history
    ...     target_col='uplink_bytes'  # Predict next value
    ... )
    >>> print(X.shape)  # (n_samples, 48, 2)
    """
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    sequences = []
    targets = []
    
    # Sort by entity and time
    sort_cols = entity_col + [time_col]
    df = df.sort_values(sort_cols)
    
    # Process each entity separately
    for _, group in df.groupby(entity_col):
        values = group[feature_cols].values
        n_samples = len(values)
        
        if n_samples < seq_length:
            continue  # Skip entities with insufficient data
        
        # Create sequences with sliding window
        for i in range(0, n_samples - seq_length, step):
            seq = values[i:i + seq_length]
            sequences.append(seq)
            
            if target_col is not None:
                # Target is the value at the end of the sequence (or next step)
                if i + seq_length < n_samples:
                    target_idx = i + seq_length
                    targets.append(group[target_col].iloc[target_idx])
                else:
                    targets.append(group[target_col].iloc[-1])
    
    X = np.array(sequences)
    
    if target_col is not None:
        y = np.array(targets)
        return X, y
    
    return X


def extract_sequences_with_metadata(
    df: pd.DataFrame,
    entity_col: Union[str, List[str]],
    feature_cols: List[str],
    seq_length: int,
    time_col: str = 'timestamp',
    step: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract sequences and return metadata about each sequence.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dense time-series DataFrame.
    entity_col : str or list of str
        Entity identifier column(s).
    feature_cols : list of str
        Feature columns.
    seq_length : int
        Sequence length.
    time_col : str
        Time column.
    step : int
        Stride between sequences.
        
    Returns
    -------
    tuple of (np.ndarray, pd.DataFrame)
        X: sequences with shape (n_sequences, seq_length, n_features)
        metadata: DataFrame with entity and time info for each sequence
        
    Example
    -------
    >>> X, meta = extract_sequences_with_metadata(df, 'imsi', ['uplink'], 48)
    >>> # meta contains: imsi, seq_start_time, seq_end_time for each sequence
    """
    if isinstance(entity_col, str):
        entity_col = [entity_col]
    
    sequences = []
    meta_records = []
    
    sort_cols = entity_col + [time_col]
    df = df.sort_values(sort_cols)
    
    for entity_key, group in df.groupby(entity_col):
        if not isinstance(entity_key, tuple):
            entity_key = (entity_key,)
            
        values = group[feature_cols].values
        times = group[time_col].values
        n_samples = len(values)
        
        if n_samples < seq_length:
            continue
        
        for i in range(0, n_samples - seq_length + 1, step):
            seq = values[i:i + seq_length]
            sequences.append(seq)
            
            meta_record = dict(zip(entity_col, entity_key))
            meta_record['seq_start_time'] = times[i]
            meta_record['seq_end_time'] = times[i + seq_length - 1]
            meta_record['seq_index'] = i
            meta_records.append(meta_record)
    
    X = np.array(sequences)
    metadata = pd.DataFrame(meta_records)
    
    return X, metadata
