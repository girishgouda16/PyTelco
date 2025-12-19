"""
PyTelco Demo: Real Data Science Workflows
==========================================
This demonstrates how a data scientist would ACTUALLY use PyTelco.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

# Import PyTelco - like sklearn
from pytelco import (
    # Core utilities
    to_dense_timeseries,
    add_lags,
    add_rolling,
    extract_sequences,
)
from pytelco.temporal import compute_slope, compute_velocity
from pytelco.features import compute_cdr_metrics, compute_gtpu_metrics, compute_sip_metrics
from pytelco.io import load_sip, load_gtpu, load_cdr


def demo_cdr_workflow():
    """
    Workflow 1: CDR Analysis for Churn Prediction
    
    Goal: Create features for a churn prediction model
    """
    print("\n" + "="*60)
    print("WORKFLOW 1: CDR Analysis for Churn Prediction")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading CDR data...")
    cdr_df = load_cdr("data/cdr/")
    print(f"  Loaded {len(cdr_df)} raw CDR records")
    print(f"  Unique subscribers: {cdr_df['imsi'].nunique()}")
    print(f"  Date range: {cdr_df['timestamp'].min()} to {cdr_df['timestamp'].max()}")
    
    # Step 2: Convert to dense time-series (THE KEY STEP)
    print("\n[Step 2] Converting sparse events to dense time-series...")
    dense_df = to_dense_timeseries(
        cdr_df,
        entity_cols=['imsi'],
        time_col='timestamp',
        value_cols=['uplink_bytes', 'downlink_bytes'],
        freq='1D',  # Daily buckets
        fill_value=0  # Zero-fill inactive days
    )
    print(f"  Dense shape: {dense_df.shape}")
    print(f"  Sample (first user, first 5 days):")
    sample_imsi = dense_df['imsi'].iloc[0]
    print(dense_df[dense_df['imsi'] == sample_imsi].head())
    
    # Step 3: Add lag features
    print("\n[Step 3] Adding lag features (t-1, t-7)...")
    dense_df = add_lags(
        dense_df, 
        cols=['uplink_bytes', 'downlink_bytes'],
        lags=[1, 7],  # Yesterday and last week
        entity_col='imsi'
    )
    print(f"  New columns: {[c for c in dense_df.columns if 'lag' in c]}")
    
    # Step 4: Add rolling statistics
    print("\n[Step 4] Adding rolling statistics (7-day, 30-day)...")
    dense_df = add_rolling(
        dense_df,
        cols=['uplink_bytes'],
        windows=[7, 30],
        funcs=['mean', 'std'],
        entity_col='imsi'
    )
    print(f"  New columns: {[c for c in dense_df.columns if 'rolling' in c]}")
    
    # Step 5: Add trend features
    print("\n[Step 5] Computing usage trend (slope over 7 days)...")
    dense_df = compute_slope(dense_df, col='uplink_bytes', window=7, entity_col='imsi')
    print(f"  New column: uplink_bytes_slope_7")
    
    # Show final feature set
    print("\n[Result] Final feature matrix:")
    print(f"  Shape: {dense_df.shape}")
    print(f"  Columns: {list(dense_df.columns)}")
    print("\n  Sample row:")
    print(dense_df.dropna().iloc[0])
    
    return dense_df


def demo_gtpu_sequence_workflow():
    """
    Workflow 2: GTP-U Traffic Classification (ML/DL Ready)
    
    Goal: Prepare sequences for LSTM-based device classification
    """
    print("\n" + "="*60)
    print("WORKFLOW 2: GTP-U Sequence Extraction for Deep Learning")
    print("="*60)
    
    # Step 1: Load data
    print("\n[Step 1] Loading GTP-U traffic data...")
    gtpu_df = load_gtpu("data/gtpu/")
    print(f"  Loaded {len(gtpu_df)} packets")
    print(f"  Unique sessions (TEIDs): {gtpu_df['teid'].nunique()}")
    
    # Step 2: Compute per-bucket metrics
    print("\n[Step 2] Computing traffic metrics per 5-minute bucket...")
    metrics_df = compute_gtpu_metrics(gtpu_df, entity_col='teid', freq='5min')
    print(f"  Metrics shape: {metrics_df.shape}")
    print(f"  Metrics: {list(metrics_df.columns)}")
    
    # Step 3: Convert to dense time-series
    print("\n[Step 3] Creating dense time-series grid...")
    dense_df = to_dense_timeseries(
        metrics_df,
        entity_cols=['teid'],
        time_col='timestamp',
        value_cols=['throughput_bps', 'payload_entropy', 'small_packet_ratio'],
        freq='5min',
        fill_value=0
    )
    print(f"  Dense shape: {dense_df.shape}")
    
    # Step 4: Extract sequences for LSTM
    print("\n[Step 4] Extracting sequences for deep learning...")
    feature_cols = ['throughput_bps', 'payload_entropy', 'small_packet_ratio']
    
    # Use shorter sequence for demo (data is limited)
    seq_length = min(6, len(dense_df) // dense_df['teid'].nunique())
    if seq_length < 2:
        seq_length = 2
    
    X = extract_sequences(
        dense_df,
        entity_col='teid',
        feature_cols=feature_cols,
        seq_length=seq_length,
        step=1
    )
    
    if len(X) == 0:
        print(f"  Note: Insufficient data for sequences. Need more time buckets per TEID.")
        return np.array([])
        
    print(f"  Sequence array shape: {X.shape}")
    print(f"  -> {X.shape[0]} sequences")
    print(f"  -> {X.shape[1]} time steps each")
    print(f"  -> {X.shape[2]} features per step")
    
    print("\n  Ready for:")
    print("    model = LSTM(units=64, input_shape=(12, 3))")
    print("    model.fit(X, y_labels)")
    
    return X


def demo_sip_analysis():
    """
    Workflow 3: SIP Signaling Analysis
    
    Goal: Detect anomalies in call setup patterns
    """
    print("\n" + "="*60)
    print("WORKFLOW 3: SIP Signaling Quality Analysis")
    print("="*60)
    
    # Step 1: Load SIP data
    print("\n[Step 1] Loading SIP signaling data...")
    sip_df = load_sip("data/sip/")
    print(f"  Loaded {len(sip_df)} SIP events")
    print(f"  Unique sources: {sip_df['contact_uri'].nunique()}")
    
    # Step 2: Compute SIP metrics
    print("\n[Step 2] Computing SIP quality metrics...")
    from pytelco.features.sip import compute_sip_metrics, compute_inter_arrival_time
    
    metrics_df = compute_sip_metrics(sip_df, entity_col='contact_uri', freq='10min')
    print(f"  Metrics shape: {metrics_df.shape}")
    print(metrics_df.head())
    
    # Step 3: Compute inter-arrival time (flood detection)
    print("\n[Step 3] Computing inter-arrival time statistics...")
    iat_df = compute_inter_arrival_time(sip_df, entity_col='contact_uri')
    print("  Top 5 sources by event count:")
    print(iat_df.nlargest(5, 'event_count'))
    
    # Identify potential attackers (very low IAT std = automated)
    print("\n[Insight] Potential automated sources (low IAT variance):")
    suspicious = iat_df[iat_df['iat_std_sec'] < 0.1]
    print(suspicious)


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# PyTelco v0.2.0 - The Telco Data Science Toolkit")
    print("#"*60)
    
    # Run all workflows
    feature_df = demo_cdr_workflow()
    X_sequences = demo_gtpu_sequence_workflow()
    demo_sip_analysis()
    
    print("\n" + "="*60)
    print("SUMMARY: PyTelco enables sklearn-like workflows for telco data")
    print("="*60)
    print("""
Key Functions Used:
    1. to_dense_timeseries() - Sparse events -> Dense grid
    2. add_lags()            - Add historical features
    3. add_rolling()         - Add rolling statistics
    4. compute_slope()       - Trend direction
    5. extract_sequences()   - DL-ready sequences
    6. compute_*_metrics()   - Domain-specific features
    """)
