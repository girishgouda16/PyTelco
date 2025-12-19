"""
PyTelco: The Telco Data Science Toolkit
========================================
A sklearn-style library for telecommunications data analysis.

Modules:
    preprocessing - Data preparation (densification, validation, cleaning)
    temporal      - Time-series operations (lags, rolling, sequences)
    features      - Domain-specific feature extractors
    io            - Data loaders for SIP, GTP-U, CDR
"""

__version__ = "0.2.2"

# Core utilities (most used)
from .preprocessing.time_series import (
    to_dense_timeseries,
    align_to_grid,
)

from .preprocessing.cleaning import (
    fill_missing,
    clip_outliers,
    remove_inactive,
)

from .preprocessing.validation import (
    validate_schema,
    get_schema_report,
)

from .preprocessing.feature_utils import (
    get_feature_names,
    describe_features,
    FeaturePipeline,
)

from .temporal.lags import (
    add_lags,
    add_rolling,
    add_diff,
)

from .temporal.sequences import (
    extract_sequences,
)

from .temporal.trends import (
    compute_slope,
    compute_velocity,
)

# Convenience imports
from . import preprocessing
from . import temporal
from . import features
from . import io
