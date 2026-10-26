"""
HYDROTWIN-AI: Scenario & Experiment Laboratory
Pre-configured operational and stress scenarios for data-center cooling water.
"""

from typing import Dict, Any

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Normal Operation": {
        "description": "Standard steady-state operation with municipal freshwater at nominal data-center IT thermal load (10 MW).",
        "config_overrides": {
            "source_name": "Freshwater",
            "thermal_load": 10.0,
            "evaporation_factor": 1.5,
            "makeup_capacity_m3_h": 25.0,
            "blowdown_capacity_m3_h": 18.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
    "High Thermal Load": {
        "description": "Peak AI computing / heat wave conditions (22 MW thermal load), driving high evaporation and rapid salt accumulation.",
        "config_overrides": {
            "source_name": "Freshwater",
            "thermal_load": 22.0,
            "evaporation_factor": 1.6,
            "makeup_capacity_m3_h": 45.0,
            "blowdown_capacity_m3_h": 25.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
    "Poor Water Quality": {
        "description": "High-TDS / reclaimed secondary effluent source water with elevated initial chloride, sulfate, and conductivity.",
        "config_overrides": {
            "source_name": "Reclaimed water (high TDS)",
            "thermal_load": 12.0,
            "evaporation_factor": 1.5,
            "makeup_capacity_m3_h": 25.0,
            "blowdown_capacity_m3_h": 22.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
    "Restricted Makeup Inflow": {
        "description": "Severe supply-side restriction (e.g. municipal drought rationing or booster pump fault) capping makeup to 10 m³/h.",
        "config_overrides": {
            "source_name": "Freshwater",
            "thermal_load": 10.0,
            "evaporation_factor": 1.5,
            "makeup_capacity_m3_h": 10.0,
            "blowdown_capacity_m3_h": 15.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
    "Restricted Blowdown Discharge": {
        "description": "Discharge sewer permit limitation or wastewater treatment bottleneck limiting blowdown to only 6 m³/h.",
        "config_overrides": {
            "source_name": "Freshwater",
            "thermal_load": 12.0,
            "evaporation_factor": 1.5,
            "makeup_capacity_m3_h": 25.0,
            "blowdown_capacity_m3_h": 6.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
    "Combined Multiple Stress": {
        "description": "Multi-stress emergency scenario: high thermal load (18 MW), poor water quality source, and restricted blowdown (8 m³/h).",
        "config_overrides": {
            "source_name": "Reclaimed water (high TDS)",
            "thermal_load": 18.0,
            "evaporation_factor": 1.6,
            "makeup_capacity_m3_h": 35.0,
            "blowdown_capacity_m3_h": 8.0,
            "ec_low": 1150.0,
            "ec_high": 1290.0,
            "wqri_trigger": 0.60,
            "makeup_mode": "Automatic",
        },
    },
}


def get_scenario_config(scenario_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a merged configuration dictionary for the selected scenario."""
    cfg = dict(base_config)
    if scenario_name in SCENARIOS:
        cfg.update(SCENARIOS[scenario_name]["config_overrides"])
    return cfg
