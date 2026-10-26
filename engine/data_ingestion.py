"""
HYDROTWIN-AI: Data Ingestion & Schema Validation Interface
Enables flexible ingestion of external CSV time-series datasets or real physical sensor telemetry,
performing schema validation, boundary checks, and units verification.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np


# Standard Expected Schema for HydroTwin Data Ingestion
EXPECTED_COLUMNS: Dict[str, Dict[str, Any]] = {
    "time_h": {
        "required": True,
        "dtype": "float",
        "min": 0.0,
        "max": 10000.0,
        "description": "Simulation or observation elapsed time in hours",
    },
    "basin_volume_m3": {
        "required": True,
        "dtype": "float",
        "min": 0.0,
        "max": 50000.0,
        "description": "Recirculating basin water volume in m³",
    },
    "EC_estimate_uScm": {
        "required": True,
        "dtype": "float",
        "min": 0.0,
        "max": 20000.0,
        "description": "Electrical conductivity estimate or measurement in µS/cm",
    },
    "pH": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 14.0,
        "description": "Water pH (acidity / alkalinity)",
    },
    "TDS_mg_l": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 5000.0,
        "description": "Total Dissolved Solids in mg/L",
    },
    "hardness_mg_l": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 5000.0,
        "description": "Total Hardness as CaCO3 in mg/L",
    },
    "temperature_C": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 100.0,
        "description": "Water temperature in degrees Celsius",
    },
    "WQRI_provisional": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Provisional Water Quality Risk Index in [0, 1]",
    },
    "WQRI_composite": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Composite Water Quality Risk Index in [0, 1]",
    },
    "SRI_scaling_index": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Scaling Risk Index in [0, 1]",
    },
    "CRI_corrosion_index": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 1.0,
        "description": "Corrosion Risk Index in [0, 1]",
    },
    "LSI": {
        "required": False,
        "dtype": "float",
        "min": -5.0,
        "max": 5.0,
        "description": "Langelier Saturation Index",
    },
    "cooling_efficiency_pct": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 100.0,
        "description": "Derated heat-exchanger cooling efficiency in percent",
    },
    "Cl_mg_l": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 10000.0,
        "description": "Chloride concentration in mg/L",
    },
    "SO4_mg_l": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 10000.0,
        "description": "Sulfate concentration in mg/L",
    },
    "Ca_mg_l": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 10000.0,
        "description": "Calcium concentration in mg/L",
    },
    "makeup_m3_h": {
        "required": True,
        "dtype": "float",
        "min": 0.0,
        "max": 1000.0,
        "description": "Makeup water inflow rate in m³/h",
    },
    "blowdown_m3_h": {
        "required": True,
        "dtype": "float",
        "min": 0.0,
        "max": 1000.0,
        "description": "Blowdown bleed outflow rate in m³/h",
    },
    "evaporation_m3_h": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 1000.0,
        "description": "Evaporative water loss rate in m³/h",
    },
    "thermal_load_MW": {
        "required": False,
        "dtype": "float",
        "min": 0.0,
        "max": 500.0,
        "description": "IT thermal load dissipation in MW",
    },
}


class ValidationResult:
    """Holds outcome and diagnostic messages from schema verification."""
    def __init__(self, is_valid: bool, errors: List[str], warnings: List[str], summary: Dict[str, Any]):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings
        self.summary = summary

    def __repr__(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return f"<ValidationResult status={status} errors={len(self.errors)} warnings={len(self.warnings)}>"


def validate_time_series_dataset(df: pd.DataFrame) -> ValidationResult:
    """
    Validates an ingested DataFrame against the HYDROTWIN-AI time-series schema.
    
    Checks:
    1. Presence of all mandatory required columns
    2. Numerical validity (no infs, NaNs handled)
    3. Monotonic non-decreasing time steps
    4. Physical bounds (e.g., non-negative volume, flow rates, conductivity)
    """
    errors: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}
    
    if df.empty:
        errors.append("Dataset is empty; zero rows found.")
        return ValidationResult(False, errors, warnings, summary)
        
    summary["row_count"] = len(df)
    summary["column_count"] = len(df.columns)
    
    # Check required columns
    missing_required = [
        col for col, meta in EXPECTED_COLUMNS.items()
        if meta["required"] and col not in df.columns
    ]
    if missing_required:
        errors.append(f"Missing mandatory required column(s): {missing_required}")
        
    # Check optional columns
    missing_optional = [
        col for col, meta in EXPECTED_COLUMNS.items()
        if not meta["required"] and col not in df.columns
    ]
    if missing_optional:
        warnings.append(f"Optional column(s) not present (will use default proxies if needed): {missing_optional}")
        
    # Check for NaN / Inf in required columns
    for col in EXPECTED_COLUMNS.keys():
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    warnings.append(f"Column '{col}' was converted to numeric.")
                except Exception:
                    errors.append(f"Column '{col}' must contain numeric data.")
                    continue
                    
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Column '{col}' contains {nan_count} NaN or missing values.")
                
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                errors.append(f"Column '{col}' contains {inf_count} infinite values.")
                
            # Range checks
            col_min = EXPECTED_COLUMNS[col]["min"]
            col_max = EXPECTED_COLUMNS[col]["max"]
            if (df[col] < col_min).any():
                violating = (df[col] < col_min).sum()
                errors.append(f"Column '{col}' contains {violating} values below physical minimum ({col_min}).")
            if (df[col] > col_max).any():
                violating = (df[col] > col_max).sum()
                warnings.append(f"Column '{col}' contains {violating} values exceeding expected normal range ({col_max}).")

    # Time monotonicity check
    if "time_h" in df.columns and len(df) > 1:
        time_diffs = df["time_h"].diff().iloc[1:]
        if (time_diffs <= 0).any():
            errors.append("Column 'time_h' is not strictly monotonically increasing.")
        summary["duration_h"] = float(df["time_h"].max() - df["time_h"].min())
        summary["median_dt_h"] = float(time_diffs.median())
        
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings, summary)


def load_dataset_from_csv(csv_path_or_file) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Loads and validates a cooling-water dataset from a CSV path or file buffer.
    Returns (DataFrame, ValidationResult).
    """
    try:
        df = pd.read_csv(csv_path_or_file)
    except Exception as exc:
        return pd.DataFrame(), ValidationResult(False, [f"Failed to parse CSV file: {exc}"], [], {})
        
    val_res = validate_time_series_dataset(df)
    return df, val_res
