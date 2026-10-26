"""
HYDROTWIN-AI: Performance Metrics & Control Strategy Comparison
Calculates comprehensive metrics for water consumption, water quality risk,
valve wear, and operating compliance.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_simulation_metrics(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes key performance indicators (KPIs) from a simulation run.
    Integrates trapezoidal/step volume flows and measures safety limit compliance.
    """
    dt_h = float(config.get("dt_h", 0.5))
    ec_high = float(config.get("ec_high", 1290.0))
    wqri_trigger = float(config.get("wqri_trigger", 0.60))

    # Flow volumes: exclude step 0 (initial condition) for flux integrations
    active_steps = df.iloc[1:] if len(df) > 1 else df

    total_makeup = float((active_steps["makeup_m3_h"] * dt_h).sum())
    total_blowdown = float((active_steps["blowdown_m3_h"] * dt_h).sum())
    total_evaporation = float((active_steps["evaporation_m3_h"] * dt_h).sum())

    avg_ec = float(df["EC_estimate_uScm"].mean())
    max_ec = float(df["EC_estimate_uScm"].max())
    min_ec = float(df["EC_estimate_uScm"].min())

    avg_wqri = float(df["WQRI_provisional"].mean())
    max_wqri = float(df["WQRI_provisional"].max())

    # Compliance tracking
    ec_violations = (df["EC_estimate_uScm"] > ec_high).sum()
    wqri_violations = (df["WQRI_provisional"] > wqri_trigger).sum()
    any_violation = ((df["EC_estimate_uScm"] > ec_high) | (df["WQRI_provisional"] > wqri_trigger)).sum()

    total_timesteps = len(df)
    pct_within_limits = float(100.0 * (1.0 - (any_violation / max(total_timesteps, 1))))
    ec_violation_hours = float(ec_violations * dt_h)
    wqri_violation_hours = float(wqri_violations * dt_h)

    # Valve switching events (count transitions from 0 to 1)
    makeup_switches = int((df["makeup_active"].astype(int).diff().abs() == 1).sum())
    blowdown_switches = int((df["blowdown_active"].astype(int).diff().abs() == 1).sum())

    # Water efficiency: useful cooling (evaporation) vs total intake (makeup)
    water_efficiency_pct = float(100.0 * total_evaporation / max(total_makeup, 1e-9)) if total_makeup > 0 else 0.0

    # Max cycles of concentration achieved
    max_coc = float(df["cycles_of_concentration"].max()) if "cycles_of_concentration" in df.columns else 1.0

    return {
        "policy": config.get("policy", "Unknown"),
        "total_makeup_m3": total_makeup,
        "total_blowdown_m3": total_blowdown,
        "total_evaporation_m3": total_evaporation,
        "water_efficiency_pct": water_efficiency_pct,
        "avg_ec_uScm": avg_ec,
        "max_ec_uScm": max_ec,
        "min_ec_uScm": min_ec,
        "avg_wqri": avg_wqri,
        "max_wqri": max_wqri,
        "max_coc": max_coc,
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
    2. Risk-aware (EC + provisional WQRI)
    3. Water-use optimization (dynamic conservation throttling)
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
