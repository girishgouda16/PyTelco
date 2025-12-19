# PyTelco 🛰️

**The Telco Data Science Toolkit** - An sklearn-style Python library for telecommunications data analysis.

[![PyPI version](https://badge.fury.io/py/pytelco.svg)](https://pypi.org/project/pytelco/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Problem Does PyTelco Solve?

**The Problem:** Telco data (CDR, SIP, GTP-U) comes as **sparse events** - a record only exists when something happens. Machine learning models need **dense, regular time-series** with features like "what happened yesterday" or "7-day average".

**The Solution:** PyTelco provides functions to:
1. Convert sparse events → dense time-series
2. Add temporal features (lags, rolling stats, trends)
3. Extract sequences for deep learning
4. Clean and validate data

---

## 📦 Installation

```bash
pip install pytelco
```

---

## 🔧 Core Functions & Input/Output

### 1️⃣ `to_dense_timeseries()` - THE Most Important Function

**What it does:** Converts sparse event data into a regular time grid with no gaps.

**Input DataFrame (Sparse):**
```
| imsi           | timestamp           | uplink_bytes | downlink_bytes |
|----------------|---------------------|--------------|----------------|
| 310410102003   | 2024-01-01 08:23:00 | 15000        | 150000         |
| 310410102003   | 2024-01-01 14:45:00 | 8000         | 80000          |
| 310410102003   | 2024-01-03 09:12:00 | 12000        | 120000         |  ← Gap on Jan 2!
| 310410999888   | 2024-01-01 10:00:00 | 5000         | 50000          |
```

**Code:**
```python
from pytelco import to_dense_timeseries

dense_df = to_dense_timeseries(
    df,
    entity_cols=['imsi'],       # Group by this column
    time_col='timestamp',       # Your timestamp column
    value_cols=['uplink_bytes', 'downlink_bytes'],  # Columns to aggregate
    freq='1D',                  # Daily buckets
    fill_value=0                # Fill missing days with 0
)
```

**Output DataFrame (Dense):**
```
| imsi           | timestamp  | uplink_bytes | downlink_bytes |
|----------------|------------|--------------|----------------|
| 310410102003   | 2024-01-01 | 23000        | 230000         |  ← Aggregated
| 310410102003   | 2024-01-02 | 0            | 0              |  ← Zero-filled!
| 310410102003   | 2024-01-03 | 12000        | 120000         |
| 310410999888   | 2024-01-01 | 5000         | 50000          |
| 310410999888   | 2024-01-02 | 0            | 0              |  ← Zero-filled!
| 310410999888   | 2024-01-03 | 0            | 0              |  ← Zero-filled!
```

---

### 2️⃣ `add_lags()` - Add Historical Features

**What it does:** Creates columns with values from previous time steps (e.g., "yesterday's value").

**Input:** Dense DataFrame from step 1

**Code:**
```python
from pytelco import add_lags

df = add_lags(
    dense_df,
    cols=['uplink_bytes'],        # Columns to create lags for
    lags=[1, 7],                  # t-1 (yesterday), t-7 (last week)
    entity_col='imsi',            # Lags computed per user
    time_col='timestamp'
)
```

**Output:**
```
| imsi           | timestamp  | uplink_bytes | uplink_bytes_lag_1 | uplink_bytes_lag_7 |
|----------------|------------|--------------|--------------------|--------------------|
| 310410102003   | 2024-01-01 | 23000        | NaN                | NaN                |
| 310410102003   | 2024-01-02 | 0            | 23000              | NaN                |
| 310410102003   | 2024-01-03 | 12000        | 0                  | NaN                |
| 310410102003   | 2024-01-08 | 5000         | ...                | 23000              |
```

---

### 3️⃣ `add_rolling()` - Add Rolling Statistics

**What it does:** Computes statistics over a sliding window (e.g., "7-day average").

**Code:**
```python
from pytelco import add_rolling

df = add_rolling(
    df,
    cols=['uplink_bytes'],
    windows=[7, 30],              # 7-day and 30-day windows
    funcs=['mean', 'std'],        # Mean and standard deviation
    entity_col='imsi'
)
```

**Output adds columns:**
- `uplink_bytes_rolling_7_mean` - 7-day moving average
- `uplink_bytes_rolling_7_std` - 7-day standard deviation
- `uplink_bytes_rolling_30_mean` - 30-day moving average
- `uplink_bytes_rolling_30_std` - 30-day standard deviation

---

### 4️⃣ `compute_slope()` - Detect Trends

**What it does:** Computes the trend direction (is usage increasing or decreasing?).

**Code:**
```python
from pytelco import compute_slope

df = compute_slope(
    df,
    col='uplink_bytes',
    window=7,                     # Slope over 7 days
    entity_col='imsi'
)
```

**Output adds:** `uplink_bytes_slope_7`
- Positive value = usage increasing
- Negative value = usage decreasing (churn indicator!)
- Near zero = stable

---

### 5️⃣ `extract_sequences()` - For LSTM/Transformer Models

**What it does:** Creates 3D arrays for deep learning models.

**Code:**
```python
from pytelco import extract_sequences

X = extract_sequences(
    df,
    entity_col='imsi',
    feature_cols=['uplink_bytes', 'downlink_bytes'],
    seq_length=7                  # 7 days of history
)
# X.shape = (n_samples, 7, 2)
```

**Output:** NumPy array ready for `model.fit(X, y)`

---

### 6️⃣ `fill_missing()` - Handle Missing Values

**Code:**
```python
from pytelco import fill_missing

df = fill_missing(
    df,
    cols=['uplink_bytes'],
    strategy='forward',           # forward, zero, mean, interpolate
    entity_col='imsi'
)
```

---

### 7️⃣ `clip_outliers()` - Remove Extreme Values

**Code:**
```python
from pytelco import clip_outliers

df = clip_outliers(
    df,
    cols=['uplink_bytes'],
    method='percentile',          # percentile, iqr, zscore
    upper=0.99                    # Clip top 1%
)
```

---

## 🚀 Complete Workflow Example

```python
from pytelco import (
    to_dense_timeseries,
    add_lags,
    add_rolling,
    compute_slope,
    fill_missing,
    clip_outliers
)
from pytelco.io import load_cdr
from sklearn.ensemble import RandomForestClassifier

# 1. Load your data
df = load_cdr("data/cdr/")  # Or: pd.read_csv("your_data.csv")

# 2. Convert to dense time-series
dense_df = to_dense_timeseries(
    df,
    entity_cols=['imsi'],
    value_cols=['uplink_bytes', 'downlink_bytes'],
    freq='1D',
    fill_value=0
)

# 3. Add temporal features
dense_df = add_lags(dense_df, cols=['uplink_bytes'], lags=[1, 7, 30], entity_col='imsi')
dense_df = add_rolling(dense_df, cols=['uplink_bytes'], windows=[7, 30], funcs=['mean', 'std'], entity_col='imsi')
dense_df = compute_slope(dense_df, col='uplink_bytes', window=7, entity_col='imsi')

# 4. Clean data
dense_df = fill_missing(dense_df, strategy='zero')
dense_df = clip_outliers(dense_df, cols=['uplink_bytes'], upper=0.99)

# 5. Train ML model
feature_cols = [
    'uplink_bytes', 
    'uplink_bytes_lag_1', 
    'uplink_bytes_lag_7',
    'uplink_bytes_rolling_7_mean',
    'uplink_bytes_slope_7'
]
X = dense_df[feature_cols].dropna().values
y = get_labels(dense_df)  # Your churn labels

model = RandomForestClassifier()
model.fit(X, y)
```

---

## 📊 Expected Input Data Formats

### CDR Data
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Event time |
| `imsi` | string | Subscriber ID |
| `imei` | string | Device ID (optional) |
| `uplink_bytes` | int | Upload volume |
| `downlink_bytes` | int | Download volume |
| `duration_sec` | int | Session duration |

### SIP Data
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Event time |
| `call_id` | string | Session ID |
| `contact_uri` | string | Source URI |
| `method` | string | INVITE, BYE, etc. |
| `status_code` | int | 200, 404, 500, etc. |

### GTP-U Data
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Packet time |
| `teid` | string | Tunnel ID |
| `payload_length` | int | Packet size |
| `inner_dest_port` | int | Destination port |

---

## 💡 Pro Tips

1. **Always use `to_dense_timeseries()` first** - This is the foundation for all other functions
2. **Choose `freq` wisely** - `'1D'` for daily churn, `'1H'` for hourly patterns, `'5min'` for packet analysis
3. **Entity isolation is automatic** - Lags and rolling stats don't "leak" between users
4. **Handle NaNs** - First rows after `add_lags()` will have NaN (no history yet)

---

## 📁 Package Structure

```
pytelco/
├── preprocessing/
│   ├── time_series.py    # to_dense_timeseries
│   ├── cleaning.py       # fill_missing, clip_outliers
│   └── validation.py     # validate_schema
├── temporal/
│   ├── lags.py           # add_lags, add_rolling, add_diff
│   ├── sequences.py      # extract_sequences
│   └── trends.py         # compute_slope
├── features/
│   ├── sip.py            # SIP-specific metrics
│   ├── gtpu.py           # GTP-U-specific metrics
│   └── cdr.py            # CDR-specific metrics
└── io/
    └── loaders.py        # Data loaders
```

---

## 🤝 Contributing

Contributions welcome! Feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
