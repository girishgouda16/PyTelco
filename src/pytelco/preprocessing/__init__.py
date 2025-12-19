"""
Preprocessing Module
====================
Data preparation utilities for telco data.
"""

from .time_series import to_dense_timeseries, align_to_grid
from .cleaning import fill_missing, clip_outliers, remove_inactive
from .validation import validate_schema, get_schema_report, ValidationResult
from .feature_utils import (
    get_feature_names, 
    describe_features, 
    save_feature_config,
    load_feature_config,
    FeaturePipeline,
)

__all__ = [
    # Time series
    "to_dense_timeseries", 
    "align_to_grid",
    # Cleaning
    "fill_missing",
    "clip_outliers",
    "remove_inactive",
    # Validation
    "validate_schema",
    "get_schema_report",
    "ValidationResult",
    # Feature utils
    "get_feature_names",
    "describe_features",
    "save_feature_config",
    "load_feature_config",
    "FeaturePipeline",
]
