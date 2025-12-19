"""
Temporal Module
===============
Time-series feature engineering utilities.
"""

from .lags import add_lags, add_rolling, add_diff
from .sequences import extract_sequences
from .trends import compute_slope, compute_velocity

__all__ = [
    "add_lags",
    "add_rolling", 
    "add_diff",
    "extract_sequences",
    "compute_slope",
    "compute_velocity",
]
