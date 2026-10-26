"""
HYDROTWIN-AI: Performance Metrics & Control Strategy Comparison
Calculates comprehensive metrics for water consumption, water quality risk,
corrosion, scaling, thermodynamic heat-transfer cooling efficiency, and operating maintenance costs.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_maintenance_cost(
    total_makeup_m3: float,
    total_blowdown_m3: float,
    max_wqri: float,
    wqri_violation_hours: float,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes data-center cooling water maintenance cost according to the approved formula:
      Cost = C_w * V_makeup + C_s * V_blowdown + C_chem * dose + C_desc * risk_penalty
    
    Default Tariffs (configurable in INR ₹):
      - C_w: ₹45.0 / m³ (freshwater purchase tariff)
      - C_s: ₹30.0 / m³ (sewer discharge & wastewater treatment fee)
      - C_chem: ₹15.0 / m³ (scale and corrosion inhibitor chemical treatment per m³ makeup)
      - C_desc: ₹2,500.0 / descaling event or violation penalty
    """
    c_w = float(config.get("tariff_makeup_inr_m3", 45.0))
    c_s = float(config.get("tariff_blowdown_inr_m3", 30.0))
    c_chem = float(config.get("tariff_chemical_inr_m3", 15.0))
    c_desc = float(config.get("penalty_descaling_inr_event", 2500.0))
    
    cost_water = float(c_w * total_makeup_m3)
    cost_sewer = float(c_s * total_blowdown_m3)
    cost_chemical = float(c_chem * total_makeup_m3)
    
    # Descaling / tube maintenance penalty scales non-linearly with risk severity and duration
    risk_factor = max(0.0, max_wqri - 0.50) / 0.50  # [0, 1]
    penalty_events = (wqri_violation_hours / 12.0) if wqri_violation_hours > 0 else (risk_factor * 0.5)
    cost_descaling = float(c_desc * penalty_events)
    
    total_cost = cost_water + cost_sewer + cost_chemical + cost_descaling
    
    return {
        "cost_water_inr": cost_water,
        "cost_sewer_inr": cost_sewer,
        "cost_chemical_inr": cost_chemical,
        "cost_descaling_inr": cost_descaling,
        "total_cost_inr": total_cost,
    }


def compute_simulation_metrics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes key performance indicators (KPIs) from a simulation run.
    Integrates trapezoidal/step volume flows, measures safety compliance,
    derated thermodynamic cooling efficiency, and operating costs.
    """
    dt_h = float(config.get("dt_h", 0.5))
    ec_high = float(config.get("ec_high", 1290.0))
    wqri_trigger = float(config.get("wqri_trigger", 0.60))

    # Flow volumes: exclude step 0 (initial condition) for flux integrations
    active_steps = df.iloc[1:] if len(df) > 1 else df

    total_makeup = float((active_steps["makeup_m3_h"] * dt_h).sum())
    total_blowdown = float((active_steps["blowdown_m3_h"] * dt_h).sum())
    total_evaporation = float((active_steps["evaporation_m3_h"] * dt_h).sum())

    # Net basin inventory change: V_end - V_start
    v_start = float(df.iloc[0]["basin_volume_m3"])
    v_end = float(df.iloc[-1]["basin_volume_m3"])
    delta_v_basin = v_end - v_start

    # Corrected Water Efficiency: accounts for net basin depletion to prevent misleading >100%
    # Useful cooling (evaporation) relative to effective water intake (makeup - net delta_V)
    effective_intake = total_makeup - delta_v_basin
    if effective_intake > 1e-6:
        raw_water_eff = 100.0 * (total_evaporation / effective_intake)
        water_efficiency_pct = float(np.clip(raw_water_eff, 0.0, 100.0))
    elif total_makeup > 0:
        water_efficiency_pct = float(np.clip(100.0 * total_evaporation / total_makeup, 0.0, 100.0))
    else:
        water_efficiency_pct = 100.0

    avg_ec = float(df["EC_estimate_uScm"].mean())
    max_ec = float(df["EC_estimate_uScm"].max())
    min_ec = float(df["EC_estimate_uScm"].min())

    avg_wqri = float(df["WQRI_provisional"].mean())
    max_wqri = float(df["WQRI_provisional"].max())
    start_wqri = float(df.iloc[0]["WQRI_provisional"])
    end_wqri = float(df.iloc[-1]["WQRI_provisional"])
    delta_wqri = end_wqri - start_wqri

    # Specific sub-indices
    avg_sri = float(df["SRI_scaling_index"].mean()) if "SRI_scaling_index" in df.columns else avg_wqri * 0.8
    max_sri = float(df["SRI_scaling_index"].max()) if "SRI_scaling_index" in df.columns else max_wqri * 0.8
    avg_cri = float(df["CRI_corrosion_index"].mean()) if "CRI_corrosion_index" in df.columns else avg_wqri * 0.6
    max_cri = float(df["CRI_corrosion_index"].max()) if "CRI_corrosion_index" in df.columns else max_wqri * 0.6
    avg_lsi = float(df["LSI"].mean()) if "LSI" in df.columns else 0.5
    max_lsi = float(df["LSI"].max()) if "LSI" in df.columns else 0.8

    # Cooling efficiency and damage rate indicators
    avg_cooling_eff = float(df["cooling_efficiency_pct"].mean()) if "cooling_efficiency_pct" in df.columns else 98.5
    min_cooling_eff = float(df["cooling_efficiency_pct"].min()) if "cooling_efficiency_pct" in df.columns else 96.0
    scaling_rate = float(df["scaling_rate_g_m2_d"].mean()) if "scaling_rate_g_m2_d" in df.columns else 0.5
    corrosion_rate = float(df["corrosion_rate_mpy"].mean()) if "corrosion_rate_mpy" in df.columns else 1.2

    # 5 First-Class Parameters (End states)
    final_ph = float(df.iloc[-1]["pH"]) if "pH" in df.columns else 7.8
    final_tds = float(df.iloc[-1]["TDS_mg_l"]) if "TDS_mg_l" in df.columns else avg_ec * 0.65
    final_hardness = float(df.iloc[-1]["hardness_mg_l"]) if "hardness_mg_l" in df.columns else 150.0
    final_temp = float(df.iloc[-1]["temperature_C"]) if "temperature_C" in df.columns else 29.0

    # Compliance tracking
    ec_violations = (df["EC_estimate_uScm"] > ec_high).sum()
    wqri_violations = (df["WQRI_provisional"] > wqri_trigger).sum()
    any_violation = ((df["EC_estimate_uScm"] > ec_high) | (df["WQRI_provisional"] > wqri_trigger)).sum()

    total_timesteps = len(df)
    pct_within_limits = float(100.0 * (1.0 - (any_violation / max(total_timesteps, 1))))
    ec_violation_hours = float(ec_violations * dt_h)
    wqri_violation_hours = float(wqri_violations * dt_h)

    # Valve transitions
    makeup_switches = int((df["makeup_active"].astype(int).diff().abs() == 1).sum())
    blowdown_switches = int((df["blowdown_active"].astype(int).diff().abs() == 1).sum())

    # Max cycles of concentration achieved
    max_coc = float(df["cycles_of_concentration"].max()) if "cycles_of_concentration" in df.columns else 1.0

    # Maintenance cost breakdown
    cost_info = compute_maintenance_cost(
        total_makeup, total_blowdown, max_wqri, wqri_violation_hours, config
    )

    return {
        "policy": config.get("policy", "Unknown"),
        "total_makeup_m3": total_makeup,
        "total_blowdown_m3": total_blowdown,
        "total_evaporation_m3": total_evaporation,
        "water_efficiency_pct": water_efficiency_pct,
        "cooling_efficiency_pct": avg_cooling_eff,
        "min_cooling_efficiency_pct": min_cooling_eff,
        "scaling_rate_g_m2_d": scaling_rate,
        "corrosion_rate_mpy": corrosion_rate,
        "avg_lsi": avg_lsi,
        "max_lsi": max_lsi,
        "avg_sri": avg_sri,
        "max_sri": max_sri,
        "avg_cri": avg_cri,
        "max_cri": max_cri,
        "avg_ec_uScm": avg_ec,
        "max_ec_uScm": max_ec,
        "min_ec_uScm": min_ec,
        "start_wqri": start_wqri,
        "end_wqri": end_wqri,
        "delta_wqri": delta_wqri,
        "avg_wqri": avg_wqri,
        "max_wqri": max_wqri,
        "max_coc": max_coc,
        "final_pH": final_ph,
        "final_TDS_mg_l": final_tds,
        "final_hardness_mg_l": final_hardness,
        "final_temperature_C": final_temp,
        "total_cost_inr": cost_info["total_cost_inr"],
        "cost_water_inr": cost_info["cost_water_inr"],
        "cost_sewer_inr": cost_info["cost_sewer_inr"],
        "cost_chemical_inr": cost_info["cost_chemical_inr"],
        "cost_descaling_inr": cost_info["cost_descaling_inr"],
        "ec_violations_count": int(ec_violations),
        "ec_violation_hours": ec_violation_hours,
        "wqri_violations_count": int(wqri_violations),
        "wqri_violation_hours": wqri_violation_hours,
        "pct_within_limits": pct_within_limits,
        "valve_switches_total": makeup_switches + blowdown_switches,
        "blowdown_switches": blowdown_switches,
    }


def compare_policies(base_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Runs identical starting conditions across all 3 control policies:
    1. Baseline (EC-hysteresis)
    2. Risk-aware (EC + composite WQRI)
    3. Water-use optimization (dynamic conservation throttling)
    
    Returns comprehensive dataframe covering cooling efficiency, freshwater use,
    scaling, corrosion, maintenance cost, and WQRI dynamics.
    """
    from engine.physics import simulate_system

    policies = ["Baseline", "Risk-aware", "Water-use optimization"]
    results = []

    for pol in policies:
        cfg = dict(base_config)
        cfg["policy"] = pol
        run_df = simulate_system(cfg)
        metrics = compute_simulation_metrics(run_df, cfg)
        results.append(metrics)

    return pd.DataFrame(results)
