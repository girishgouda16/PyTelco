"""
Schema Validation
=================
Validate input data before processing.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self):
        if self.is_valid:
            msg = "✅ Validation passed"
            if self.warnings:
                msg += f"\n⚠️ Warnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
            return msg
        else:
            return f"❌ Validation failed:\n" + "\n".join(f"  - {e}" for e in self.errors)


def validate_schema(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
    col_types: Optional[Dict[str, str]] = None,
    entity_col: Optional[Union[str, List[str]]] = None,
    time_col: Optional[str] = None,
    check_nulls: bool = True,
    check_duplicates: bool = True,
    raise_on_error: bool = False,
) -> ValidationResult:
    """
    Validate DataFrame schema before processing.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to validate.
    required_cols : list of str, optional
        Columns that must exist.
    col_types : dict, optional
        Expected types: {'col_name': 'numeric'|'string'|'datetime'|'any'}
    entity_col : str or list of str, optional
        Entity column(s) to check for consistency.
    time_col : str, optional
        Time column to validate.
    check_nulls : bool
        Check for null values in required columns.
    check_duplicates : bool
        Check for duplicate (entity, time) combinations.
    raise_on_error : bool
        If True, raise ValueError on validation failure.
        
    Returns
    -------
    ValidationResult
        Validation result with errors and warnings.
        
    Example
    -------
    >>> result = validate_schema(
    ...     df,
    ...     required_cols=['imsi', 'timestamp', 'uplink_bytes'],
    ...     col_types={'uplink_bytes': 'numeric', 'timestamp': 'datetime'},
    ...     entity_col='imsi',
    ...     time_col='timestamp'
    ... )
    >>> print(result)
    """
    errors = []
    warnings = []
    
    # Check required columns
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
    
    # Check column types
    if col_types:
        for col, expected_type in col_types.items():
            if col not in df.columns:
                continue
                
            actual = df[col].dtype
            
            if expected_type == 'numeric':
                if not pd.api.types.is_numeric_dtype(actual):
                    errors.append(f"Column '{col}' should be numeric, got {actual}")
                    
            elif expected_type == 'string':
                if not (pd.api.types.is_string_dtype(actual) or 
                        pd.api.types.is_object_dtype(actual)):
                    warnings.append(f"Column '{col}' might not be string, got {actual}")
                    
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(actual):
                    # Try to parse
                    try:
                        pd.to_datetime(df[col].head(10))
                        warnings.append(f"Column '{col}' is parseable as datetime but not typed")
                    except:
                        errors.append(f"Column '{col}' should be datetime, got {actual}")
    
    # Check time column
    if time_col and time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            try:
                pd.to_datetime(df[time_col].head(10))
                warnings.append(f"Time column '{time_col}' should be converted to datetime")
            except:
                errors.append(f"Time column '{time_col}' cannot be parsed as datetime")
    
    # Check for nulls
    if check_nulls and required_cols:
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                null_pct = null_count / len(df) * 100
                if null_pct > 0:
                    if null_pct > 50:
                        errors.append(f"Column '{col}' has {null_pct:.1f}% null values")
                    elif null_pct > 10:
                        warnings.append(f"Column '{col}' has {null_pct:.1f}% null values")
    
    # Check for duplicates
    if check_duplicates and entity_col and time_col:
        if isinstance(entity_col, str):
            entity_col = [entity_col]
        
        dup_cols = entity_col + [time_col]
        if all(c in df.columns for c in dup_cols):
            dup_count = df.duplicated(subset=dup_cols, keep=False).sum()
            if dup_count > 0:
                warnings.append(f"Found {dup_count} duplicate (entity, time) rows")
    
    # Check data size
    if len(df) == 0:
        errors.append("DataFrame is empty")
    elif len(df) < 10:
        warnings.append(f"DataFrame has only {len(df)} rows")
    
    # Build result
    is_valid = len(errors) == 0
    result = ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    if raise_on_error and not is_valid:
        raise ValueError(str(result))
    
    return result


def get_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a schema report for the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
        
    Returns
    -------
    pd.DataFrame
        Report with column stats.
    """
    report = []
    
    for col in df.columns:
        record = {
            'column': col,
            'dtype': str(df[col].dtype),
            'null_count': df[col].isna().sum(),
            'null_pct': df[col].isna().mean() * 100,
            'unique_count': df[col].nunique(),
            'sample_values': str(df[col].dropna().head(3).tolist())[:50]
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            record['min'] = df[col].min()
            record['max'] = df[col].max()
            record['mean'] = df[col].mean()
        
        report.append(record)
    
    return pd.DataFrame(report)
