# PyTelco 🛰️

**The Telco Data Science Toolkit** - An sklearn-style Python library for telecommunications data analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What is PyTelco?

PyTelco provides **low-level building blocks** for telecom data scientists to analyze CDR, SIP, and GTP-U data. Think of it as **scikit-learn for telecommunications**.

```python
from pytelco import to_dense_timeseries, add_lags, add_rolling, fill_missing

# Convert sparse CDR events to dense time-series
dense_df = to_dense_timeseries(cdr_df, entity_cols=['imsi'], freq='1D', fill_value=0)

# Add temporal features
dense_df = add_lags(dense_df, cols=['uplink_bytes'], lags=[1, 7, 30], entity_col='imsi')
dense_df = add_rolling(dense_df, cols=['uplink_bytes'], windows=[7, 30], funcs=['mean', 'std'])

# Clean data
dense_df = fill_missing(dense_df, strategy='forward', entity_col='imsi')
```

---

## 📦 Installation

```bash
# From PyPI (when published)
pip install pytelco

# From GitHub
pip install git+https://github.com/girishgouda16/PyTelco.git

# Local development
git clone https://github.com/girishgouda16/PyTelco.git
cd PyTelco
pip install -e .
```

---

## 🧰 Core Functions

### Preprocessing
| Function | Description |
|----------|-------------|
| `to_dense_timeseries()` | Convert sparse events to fixed-frequency grid |
| `fill_missing()` | Handle NaN values (zero, forward, interpolate) |
| `clip_outliers()` | Remove extreme values (percentile, IQR, zscore) |
| `validate_schema()` | Validate input data before processing |

### Temporal Features
| Function | Description |
|----------|-------------|
| `add_lags()` | Add lagged features (t-1, t-7, etc.) |
| `add_rolling()` | Add rolling statistics (mean, std, sum) |
| `add_diff()` | Add differenced features |
| `compute_slope()` | Compute trend direction |
| `extract_sequences()` | Create sequences for LSTM/Transformer models |

### Domain Features
| Function | Description |
|----------|-------------|
| `compute_sip_metrics()` | SIP signaling quality metrics |
| `compute_gtpu_metrics()` | GTP-U traffic analysis |
| `compute_cdr_metrics()` | CDR billing and usage metrics |

---

## 🚀 Quick Start

### Example 1: CDR Churn Analysis

```python
from pytelco import to_dense_timeseries, add_lags, add_rolling, compute_slope
from pytelco.io import load_cdr

# Load data
cdr_df = load_cdr("data/cdr/")

# Create dense time-series (THE KEY STEP)
dense_df = to_dense_timeseries(
    cdr_df,
    entity_cols=['imsi'],
    value_cols=['uplink_bytes', 'downlink_bytes'],
    freq='1D',
    fill_value=0
)

# Add features
dense_df = add_lags(dense_df, cols=['uplink_bytes'], lags=[1, 7, 30], entity_col='imsi')
dense_df = add_rolling(dense_df, cols=['uplink_bytes'], windows=[7, 30], funcs=['mean', 'std'])
dense_df = compute_slope(dense_df, col='uplink_bytes', window=7, entity_col='imsi')

# Ready for ML!
X = dense_df[['uplink_bytes', 'uplink_bytes_lag_1', 'uplink_bytes_rolling_7_mean']].values
```

### Example 2: Sequence Extraction for Deep Learning

```python
from pytelco import to_dense_timeseries, extract_sequences

# Create dense time-series
dense_df = to_dense_timeseries(df, entity_cols=['teid'], freq='5min')

# Extract sequences for LSTM
X = extract_sequences(
    dense_df,
    entity_col='teid',
    feature_cols=['throughput_bps', 'payload_entropy'],
    seq_length=48  # 4 hours of 5-min buckets
)

# X.shape = (n_sequences, 48, 2) - Ready for Keras/PyTorch!
```

---

## 📁 Package Structure

```
pytelco/
├── preprocessing/
│   ├── time_series.py    # to_dense_timeseries, align_to_grid
│   ├── cleaning.py       # fill_missing, clip_outliers
│   └── validation.py     # validate_schema
├── temporal/
│   ├── lags.py           # add_lags, add_rolling, add_diff
│   ├── sequences.py      # extract_sequences
│   └── trends.py         # compute_slope, compute_velocity
├── features/
│   ├── sip.py            # SIP signaling metrics
│   ├── gtpu.py           # GTP-U traffic metrics
│   └── cdr.py            # CDR billing metrics
└── io/
    └── loaders.py        # load_sip, load_gtpu, load_cdr
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
