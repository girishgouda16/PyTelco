"""
Feature Utilities
=================
Utilities for managing feature sets.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import json


def get_feature_names(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    include_pattern: Optional[str] = None,
) -> List[str]:
    """
    Get list of feature column names.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    exclude_cols : list of str, optional
        Columns to exclude (e.g., entity and time columns).
    include_pattern : str, optional
        Only include columns matching this pattern.
        
    Returns
    -------
    list of str
        Feature column names.
        
    Example
    -------
    >>> features = get_feature_names(df, exclude_cols=['imsi', 'timestamp'])
    >>> X = df[features].values
    """
    cols = df.columns.tolist()
    
    if exclude_cols:
        cols = [c for c in cols if c not in exclude_cols]
    
    if include_pattern:
        import re
        pattern = re.compile(include_pattern)
        cols = [c for c in cols if pattern.search(c)]
    
    # Only numeric columns
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols


def describe_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate summary statistics for features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features.
    feature_cols : list of str, optional
        Columns to describe. If None, uses all numeric.
        
    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = df[feature_cols].describe().T
    stats['null_pct'] = df[feature_cols].isna().mean() * 100
    stats['zero_pct'] = (df[feature_cols] == 0).mean() * 100
    
    return stats


def save_feature_config(
    config: Dict,
    filepath: str,
) -> None:
    """
    Save feature engineering configuration to JSON.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary.
    filepath : str
        Output file path.
        
    Example
    -------
    >>> config = {
    ...     'entity_cols': ['imsi'],
    ...     'time_col': 'timestamp',
    ...     'freq': '1D',
    ...     'lags': [1, 7, 30],
    ...     'rolling_windows': [7, 30],
    ...     'rolling_funcs': ['mean', 'std']
    ... }
    >>> save_feature_config(config, 'feature_config.json')
    """
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_feature_config(filepath: str) -> Dict:
    """
    Load feature engineering configuration from JSON.
    
    Parameters
    ----------
    filepath : str
        Input file path.
        
    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


class FeaturePipeline:
    """
    A configurable feature engineering pipeline.
    
    Example
    -------
    >>> pipeline = FeaturePipeline()
    >>> pipeline.add_step('dense', to_dense_timeseries, entity_cols=['imsi'], freq='1D')
    >>> pipeline.add_step('lags', add_lags, cols=['uplink_bytes'], lags=[1, 7])
    >>> result = pipeline.fit_transform(raw_df)
    """
    
    def __init__(self):
        self.steps = []
        self.feature_names_ = None
    
    def add_step(
        self, 
        name: str, 
        func: callable, 
        **kwargs
    ) -> 'FeaturePipeline':
        """Add a transformation step."""
        self.steps.append({
            'name': name,
            'func': func,
            'kwargs': kwargs
        })
        return self
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        verbose: bool = False
    ) -> pd.DataFrame:
        """Execute all steps in the pipeline."""
        result = df.copy()
        
        for step in self.steps:
            if verbose:
                print(f"Running step: {step['name']}...")
            result = step['func'](result, **step['kwargs'])
        
        self.feature_names_ = result.columns.tolist()
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after transformation."""
        return self.feature_names_
    
    def save(self, filepath: str) -> None:
        """Save pipeline configuration (note: functions not serialized)."""
        config = {
            'steps': [
                {'name': s['name'], 'kwargs': s['kwargs']}
                for s in self.steps
            ]
        }
        save_feature_config(config, filepath)
