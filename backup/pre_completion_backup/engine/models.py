"""
HYDROTWIN-AI: Physical and Chemical Data Models
Defines water source profiles, system configuration structures, and standard defaults.
"""

from typing import Dict, Any

# Standard Makeup Water Quality Profiles
# Note: Concentrations in mg/L (or mmol/L depending on reference); EC in uS/cm.
# In the prototype, these serve as baseline parameters for the mass balance equations.
WATER_PROFILES: Dict[str, Dict[str, float]] = {
    "Freshwater": {
        "EC": 345.0,
        "Cl": 0.711,
        "SO4": 0.332,
        "Ca": 1.133,
    },
    "HVAC condensate": {
        "EC": 80.0,
        "Cl": 0.030,
        "SO4": 0.020,
        "Ca": 0.020,
    },
    "Rainwater": {
        "EC": 40.0,
        "Cl": 0.020,
        "SO4": 0.010,
        "Ca": 0.020,
    },
    "Treated water": {
        "EC": 80.0,
        "Cl": 0.020,
        "SO4": 0.010,
        "Ca": 0.010,
    },
    "Reclaimed water (high TDS)": {
        "EC": 620.0,
        "Cl": 1.500,
        "SO4": 0.850,
        "Ca": 2.400,
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "source_name": "Freshwater",
    "duration_h": 48.0,
    "dt_h": 0.5,
    "basin_capacity_m3": 500.0,
    "initial_volume_m3": 400.0,
    "initial_water_age_h": 0.0,
    "thermal_load": 10.0,  # MW thermal
    "evaporation_factor": 1.5,  # m3 / (MW * h)
    "makeup_capacity_m3_h": 25.0,
    "blowdown_capacity_m3_h": 18.0,
    "ec_low": 1150.0,  # Lower EC setpoint (uS/cm) for hysteresis
    "ec_high": 1290.0,  # Upper EC setpoint (uS/cm)
    "wqri_trigger": 0.60,  # Provisional WQRI trigger [0, 1]
    "policy": "Risk-aware",  # "Baseline", "Risk-aware", or "Water-use optimization"
    "makeup_mode": "Automatic",  # "Automatic" or "Force zero makeup"
}
