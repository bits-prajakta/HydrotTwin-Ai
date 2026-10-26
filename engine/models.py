"""
HYDROTWIN-AI: Physical and Chemical Data Models
Defines water source profiles, system configuration structures, and standard defaults.

First-Class Water Quality Parameters (Approved Spec):
1. pH (acidity / alkalinity)
2. TDS (Total Dissolved Solids in mg/L)
3. Hardness (Total Hardness in mg/L as CaCO3)
4. Conductivity (Electrical Conductivity in µS/cm)
5. Temperature (Water temperature in °C)
"""

from typing import Dict, Any

# Standard Makeup Water Quality Profiles (5 Presentation Sources)
# Explicitly defines all 5 approved water quality parameters:
# - pH: dimensionless [-]
# - TDS: Total Dissolved Solids [mg/L]
# - Hardness: Total mineral hardness as CaCO3 [mg/L]
# - EC: Electrical Conductivity [µS/cm]
# - Temperature: Inflow water temperature [°C]
# Legacy ionic concentrations (Cl, SO4, Ca in mg/L) are retained for multi-species mass conservation.
WATER_PROFILES: Dict[str, Dict[str, float]] = {
    "Freshwater": {
        "pH": 7.6,
        "TDS": 224.25,        # 345 * 0.65
        "Hardness": 113.3,    # mg/L as CaCO3 (moderate)
        "EC": 345.0,
        "Temperature": 20.0,
        "Cl": 0.711,
        "SO4": 0.332,
        "Ca": 1.133,
    },
    "HVAC condensate": {
        "pH": 6.8,
        "TDS": 52.0,          # 80 * 0.65
        "Hardness": 2.0,      # demineralized / very soft
        "EC": 80.0,
        "Temperature": 18.0,
        "Cl": 0.030,
        "SO4": 0.020,
        "Ca": 0.020,
    },
    "Rainwater": {
        "pH": 6.5,
        "TDS": 26.0,          # 40 * 0.65
        "Hardness": 2.0,      # soft
        "EC": 40.0,
        "Temperature": 19.0,
        "Cl": 0.020,
        "SO4": 0.010,
        "Ca": 0.020,
    },
    "Treated water": {
        "pH": 7.2,
        "TDS": 52.0,          # 80 * 0.65
        "Hardness": 1.0,      # softened / RO polished
        "EC": 80.0,
        "Temperature": 20.0,
        "Cl": 0.020,
        "SO4": 0.010,
        "Ca": 0.010,
    },
    "Reclaimed water (high TDS)": {
        "pH": 7.9,
        "TDS": 403.0,         # 620 * 0.65
        "Hardness": 240.0,    # hard secondary effluent
        "EC": 620.0,
        "Temperature": 22.0,
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
    "thermal_load": 10.0,             # MW thermal load
    "evaporation_factor": 1.5,         # m3 / (MW * h)
    "makeup_capacity_m3_h": 25.0,
    "blowdown_capacity_m3_h": 18.0,
    "ec_low": 1150.0,                 # Lower EC setpoint (µS/cm) for hysteresis
    "ec_high": 1290.0,                # Upper EC setpoint (µS/cm)
    "wqri_trigger": 0.60,             # WQRI trigger threshold [0, 1]
    "policy": "Risk-aware",           # "Baseline", "Risk-aware", or "Water-use optimization"
    "makeup_mode": "Automatic",       # "Automatic" or "Force zero makeup"
    "ambient_temp_c": 25.0,           # Ambient dry bulb / wet bulb reference temperature (°C)
    "tds_factor": 0.65,               # Ratio TDS (mg/L) / EC (µS/cm)
    # Economic cost model tariffs (INR - Indian Rupees)
    "tariff_makeup_inr_m3": 45.0,     # Freshwater intake municipal tariff (₹/m³)
    "tariff_blowdown_inr_m3": 30.0,   # Sewer discharge & effluent treatment fee (₹/m³)
    "tariff_chemical_inr_m3": 15.0,   # Scale & corrosion chemical inhibitor dosing cost (₹/m³ makeup)
    "penalty_descaling_inr_event": 2500.0, # Tube bundle cleaning / descaling risk penalty (₹/event)
}
