"""
HYDROTWIN-AI: Physical Simulation Engine
Implements mass balances, dissolved species kinetics, 5-parameter water chemistry,
Langelier Saturation Index (LSI), Scaling/Corrosion Risk Indices (SRI/CRI),
and composite WQRI tracking for recirculating data-center cooling water.

Approved Specification Parameters (GNIMS MCA Baseline):
1. pH (acidity / alkalinity)
2. TDS (Total Dissolved Solids in mg/L)
3. Hardness (Total Hardness as CaCO3 in mg/L)
4. Conductivity (Electrical Conductivity in µS/cm)
5. Temperature (Water temperature in °C)
"""

from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from engine.models import WATER_PROFILES, DEFAULT_CONFIG


def calculate_ec(chloride_mg_l: float, source_name: str) -> float:
    """
    Illustrative chloride-based electrical conductivity (EC) proxy.
    
    Disclaimer:
    Conductivity in real industrial water depends on total dissolved solids (TDS),
    ionic strength, speciation, and temperature.
    """
    if source_name not in WATER_PROFILES:
        source_name = "Freshwater"
    source = WATER_PROFILES[source_name]
    source_chloride = max(float(source["Cl"]), 1e-9)

    estimated_ec = (
        float(source["EC"])
        * max(float(chloride_mg_l), 0.0)
        / source_chloride
    )
    return max(float(estimated_ec), 0.0)


def calculate_lsi(
    ph: float,
    tds: float,
    hardness: float,
    temp_c: float = 29.0,
    alkalinity: float = 120.0,
) -> float:
    """
    Computes the Langelier Saturation Index (LSI) for calcium carbonate scaling/corrosion tendency.
    
    LSI = pH - pHs
    where pHs = (9.3 + A + B) - (C + D)
      A = (log10(TDS) - 1) / 10
      B = -13.12 * log10(temp_c + 273.15) + 34.55
      C = log10(Hardness) - 0.4
      D = log10(Alkalinity)
    
    Interpretation:
      LSI > 0: Supersaturated with CaCO3 -> Scaling tendency
      LSI < 0: Undersaturated -> Corrosive tendency
      LSI ~ 0: Chemically balanced
    """
    safe_tds = max(float(tds), 1.0)
    safe_temp = max(float(temp_c), 5.0)
    safe_hard = max(float(hardness), 1.0)
    safe_alk = max(float(alkalinity), 1.0)
    
    a = (np.log10(safe_tds) - 1.0) / 10.0
    b = -13.12 * np.log10(safe_temp + 273.15) + 34.55
    c = np.log10(safe_hard) - 0.4
    d = np.log10(safe_alk)
    
    phs = (9.3 + a + b) - (c + d)
    lsi = float(ph - phs)
    return float(np.clip(lsi, -4.0, 4.0))


def calculate_sri(
    hardness: float,
    ph: float,
    tds: float,
    lsi: Optional[float] = None,
    limits: Optional[Dict[str, float]] = None,
) -> float:
    """
    Scaling Risk Index (SRI) [0, 1].
    Driven by mineral hardness (CaCO3), alkaline pH (>7.5), high TDS, and positive LSI.
    """
    ref_hard = 300.0 if not limits else limits.get("hardness", 300.0)
    ref_tds = 1500.0 if not limits else limits.get("tds", 1500.0)
    
    if lsi is None:
        lsi = calculate_lsi(ph, tds, hardness)
        
    c_hard = max(float(hardness), 0.0) / max(float(ref_hard), 1e-6)
    c_ph = max(float(ph) - 7.0, 0.0) / 2.0  # pH 7.0 = 0 risk, pH 9.0 = 1.0
    c_tds = max(float(tds), 0.0) / max(float(ref_tds), 1e-6)
    c_lsi = max(float(lsi), 0.0) / 2.0      # LSI +2.0 = severe scaling
    
    sri = 0.40 * c_hard + 0.25 * c_ph + 0.20 * c_tds + 0.15 * c_lsi
    return float(np.clip(sri, 0.0, 1.0))


def calculate_cri(
    ph: float,
    ec: float,
    chloride: float = 0.0,
    lsi: Optional[float] = None,
    limits: Optional[Dict[str, float]] = None,
) -> float:
    """
    Corrosion Risk Index (CRI) [0, 1].
    Driven by acidic pH (<7.8), high electrical conductivity/dissolved ions,
    chloride pitting potential, and negative LSI.
    """
    ref_ec = 2200.0 if not limits else limits.get("ec", 2200.0)
    ref_cl = 4.2 if not limits else limits.get("cl", 4.2)
    
    c_ph = max(7.8 - float(ph), 0.0) / 1.8  # pH 7.8 = 0, pH 6.0 = 1.0
    c_ec = max(float(ec), 0.0) / max(float(ref_ec), 1e-6)
    c_cl = max(float(chloride), 0.0) / max(float(ref_cl), 1e-6)
    c_lsi = max(-float(lsi if lsi is not None else 0.0), 0.0) / 2.0
    
    cri = 0.35 * c_ph + 0.35 * c_ec + 0.20 * c_cl + 0.10 * c_lsi
    return float(np.clip(cri, 0.0, 1.0))


def calculate_wqri(
    *args,
    ph: Optional[float] = None,
    tds: Optional[float] = None,
    ec: Optional[float] = None,
    hardness: Optional[float] = None,
    chloride: float = 0.0,
    sulfate: float = 0.0,
    calcium: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
    limits: Optional[Dict[str, float]] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """
    Water Quality Risk Index (WQRI) [0, 1].
    Approved Specification Formulation (GNIMS MCA Presentation Baseline).
    
    Inputs (Approved 4 Parameters):
      - pH: Water acidity/alkalinity
      - TDS: Total Dissolved Solids [mg/L]
      - Conductivity: Electrical Conductivity [µS/cm]
      - Hardness: Total mineral hardness as CaCO3 [mg/L]
    
    Optional Supplementary Terms:
      - Chloride [mg/L] (pitting corrosion risk proxy)
      - Sulfate [mg/L] (acidic corrosion / scaling proxy)
    
    Also provides sub-indices:
      - SRI (Scaling Risk Index)
      - CRI (Corrosion Risk Index)
      - LSI (Langelier Saturation Index)
      
    Backward-compatibility:
      If called positionally with legacy signature (ec, chloride, sulfate, calcium),
      the function automatically adapts without breaking existing callers or tests.
    """
    # Detect signature mode
    if len(args) >= 1:
        # Check if first positional arg is EC (legacy) or pH (new)
        first_arg = float(args[0])
        if first_arg > 14.0:
            # Legacy signature: calculate_wqri(ec, chloride, sulfate, calcium=0, weights=None, limits=None)
            ec_val = first_arg
            cl_val = float(args[1]) if len(args) > 1 else chloride
            so4_val = float(args[2]) if len(args) > 2 else sulfate
            ca_val = float(args[3]) if len(args) > 3 else calcium
            if len(args) > 4 and isinstance(args[4], dict):
                weights = args[4]
            if len(args) > 5 and isinstance(args[5], dict):
                limits = args[5]
                
            tds_val = ec_val * 0.65
            hard_val = max(ca_val * 100.0, 10.0)
            ph_val = 7.6 + 0.4 * min(ec_val / 1200.0, 1.5)
        else:
            # New specification signature: calculate_wqri(ph, tds, ec, hardness, ...)
            ph_val = first_arg
            tds_val = float(args[1]) if len(args) > 1 else (tds if tds is not None else ph_val * 0.65)
            ec_val = float(args[2]) if len(args) > 2 else (ec if ec is not None else tds_val / 0.65)
            hard_val = float(args[3]) if len(args) > 3 else (hardness if hardness is not None else 100.0)
            cl_val = float(args[4]) if len(args) > 4 else chloride
            so4_val = float(args[5]) if len(args) > 5 else sulfate
            ca_val = float(args[6]) if len(args) > 6 else (calcium if calcium else hard_val / 100.0)
    else:
        # Keyword call
        ph_val = float(ph) if ph is not None else 7.6
        ec_val = float(ec) if ec is not None else 345.0
        tds_val = float(tds) if tds is not None else (ec_val * 0.65)
        hard_val = float(hardness) if hardness is not None else max(calcium * 100.0, 100.0)
        cl_val = float(chloride)
        so4_val = float(sulfate)
        ca_val = float(calcium) if calcium else (hard_val / 100.0)

    ref_limits = {
        "ph_low": 7.0,
        "ph_high": 8.5,
        "tds": 1500.0,      # mg/L upper industrial cooling threshold
        "ec": 2200.0,       # µS/cm upper threshold
        "hardness": 300.0,  # mg/L CaCO3 threshold
        "cl": 4.2,          # mg/L supplementary reference
        "so4": 3.4,         # mg/L supplementary reference
        "ca": 5.0,          # mg/L legacy reference
    }
    if limits:
        ref_limits.update(limits)

    # Normalized component risk scores
    # pH risk: penalizes deviation outside safe window [7.2, 8.2]
    if ph_val < 7.2:
        c_ph = (7.2 - ph_val) / 1.2
    elif ph_val > 8.2:
        c_ph = (ph_val - 8.2) / 1.0
    else:
        c_ph = 0.0
    c_ph = float(np.clip(c_ph, 0.0, 1.0))

    c_tds = max(float(tds_val), 0.0) / max(float(ref_limits["tds"]), 1e-6)
    c_ec = max(float(ec_val), 0.0) / max(float(ref_limits["ec"]), 1e-6)
    c_hard = max(float(hard_val), 0.0) / max(float(ref_limits["hardness"]), 1e-6)

    # Optional supplementary ions
    c_cl = max(float(cl_val), 0.0) / max(float(ref_limits["cl"]), 1e-6)
    c_so4 = max(float(so4_val), 0.0) / max(float(ref_limits["so4"]), 1e-6)
    c_ca = max(float(ca_val), 0.0) / max(float(ref_limits["ca"]), 1e-6)

    # Configurable weights
    if weights is not None:
        # Check if caller used legacy keys ("ec", "cl", "so4", "ca")
        if "cl" in weights and "tds" not in weights:
            w_ec = float(weights.get("ec", 0.35))
            w_cl = float(weights.get("cl", 0.30))
            w_so4 = float(weights.get("so4", 0.20))
            w_ca = float(weights.get("ca", 0.15))
            w_sum = max(w_ec + w_cl + w_so4 + w_ca, 1e-6)
            wqri_val = (w_ec * c_ec + w_cl * c_cl + w_so4 * c_so4 + w_ca * c_ca) / w_sum
            wqri_val = float(np.clip(wqri_val, 0.0, 1.0))
        else:
            w_ph = float(weights.get("ph", 0.20))
            w_tds = float(weights.get("tds", 0.25))
            w_ec = float(weights.get("ec", 0.30))
            w_hard = float(weights.get("hardness", 0.25))
            w_sum = max(w_ph + w_tds + w_ec + w_hard, 1e-6)
            wqri_val = (w_ph * c_ph + w_tds * c_tds + w_ec * c_ec + w_hard * c_hard) / w_sum
            wqri_val = float(np.clip(wqri_val, 0.0, 1.0))
    else:
        # Standard balanced 4-parameter formulation
        wqri_val = 0.20 * c_ph + 0.25 * c_tds + 0.30 * c_ec + 0.25 * c_hard
        wqri_val = float(np.clip(wqri_val, 0.0, 1.0))

    lsi_val = calculate_lsi(ph_val, tds_val, hard_val)
    sri_val = calculate_sri(hard_val, ph_val, tds_val, lsi=lsi_val, limits=ref_limits)
    cri_val = calculate_cri(ph_val, ec_val, chloride=cl_val, lsi=lsi_val, limits=ref_limits)

    if return_details:
        return {
            "wqri": wqri_val,
            "sri": sri_val,
            "cri": cri_val,
            "lsi": lsi_val,
            "alarm_status": "CRITICAL" if wqri_val >= 0.60 else ("ELEVATED" if wqri_val >= 0.40 else "NORMAL"),
            "components": {
                "pH": ph_val,
                "TDS_mg_l": tds_val,
                "EC_uScm": ec_val,
                "Hardness_mg_l": hard_val,
                "Chloride_mg_l": cl_val,
                "Sulfate_mg_l": so4_val,
            }
        }

    return wqri_val


def validate_config(config: Dict[str, Any]) -> None:
    """Validates configuration parameters before running the simulation."""
    if float(config.get("duration_h", 0)) <= 0:
        raise ValueError("Simulation duration must be greater than zero.")
    if float(config.get("dt_h", 0)) <= 0:
        raise ValueError("Simulation time step dt must be greater than zero.")
    if float(config.get("basin_capacity_m3", 0)) <= 0:
        raise ValueError("Basin capacity must be strictly positive.")
    
    cap = float(config["basin_capacity_m3"])
    v0 = float(config.get("initial_volume_m3", 0))
    if not (0 < v0 <= cap):
        raise ValueError(f"Initial volume ({v0} m³) must be > 0 and <= capacity ({cap} m³).")

    ec_low = float(config.get("ec_low", 0))
    ec_high = float(config.get("ec_high", 0))
    if ec_low >= ec_high:
        raise ValueError(f"Lower EC reference ({ec_low}) must be less than upper setpoint ({ec_high}).")

    if float(config.get("makeup_capacity_m3_h", 0)) < 0 or float(config.get("blowdown_capacity_m3_h", 0)) < 0:
        raise ValueError("Flow capacities cannot be negative.")


def simulate_system(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Simulates a well-mixed recirculating cooling basin using mass balances
    coupled to all 5 approved water-quality parameters:
    1. pH (shifts dynamically with CO2 stripping / cycles of concentration)
    2. TDS (tracked via dissolved solids accumulation)
    3. Hardness (mineral scaling proxy from calcium mass balance)
    4. Electrical Conductivity (EC proxy)
    5. Basin Temperature (dynamic thermal equilibrium)
    
    Also computes:
    - LSI (Langelier Saturation Index)
    - SRI (Scaling Risk Index)
    - CRI (Corrosion Risk Index)
    - Cooling heat-exchanger derated efficiency
    - Scaling deposition rate (g/m²/day) & corrosion rate (mpy)
    """
    validate_config(config)

    source_name = config.get("source_name", "Freshwater")
    if source_name not in WATER_PROFILES:
        source_name = "Freshwater"
    source = WATER_PROFILES[source_name]

    duration_h = float(config["duration_h"])
    dt_h = float(config["dt_h"])
    basin_capacity = float(config["basin_capacity_m3"])
    initial_volume = float(config["initial_volume_m3"])
    thermal_load = float(config.get("thermal_load", 10.0))
    evaporation_factor = float(config.get("evaporation_factor", 1.5))
    makeup_capacity = float(config.get("makeup_capacity_m3_h", 25.0))
    blowdown_capacity = float(config.get("blowdown_capacity_m3_h", 18.0))
    ec_low = float(config.get("ec_low", 1150.0))
    ec_high = float(config.get("ec_high", 1290.0))
    wqri_trigger = float(config.get("wqri_trigger", 0.60))
    policy = config.get("policy", "Risk-aware")
    makeup_mode = config.get("makeup_mode", "Automatic")
    ambient_temp = float(config.get("ambient_temp_c", 25.0))
    tds_factor = float(config.get("tds_factor", 0.65))

    n_steps = int(np.ceil(duration_h / dt_h))
    time_values = np.minimum(
        np.arange(n_steps + 1, dtype=float) * dt_h,
        duration_h,
    )

    # Evaporation rate (m³/h)
    evaporation_rate = max(thermal_load * evaporation_factor, 0.0)

    # State variables
    volume = initial_volume
    # Mass of species in mg: concentration (mg/L) * volume (m³) * 1000 (L/m³)
    chloride_mass = max(float(source["Cl"]), 0.0) * volume * 1000.0
    sulfate_mass = max(float(source["SO4"]), 0.0) * volume * 1000.0
    calcium_mass = max(float(source["Ca"]), 0.0) * volume * 1000.0

    age_inventory = 0.0
    blowdown_latched = False  # For hysteresis controller tracking

    records = []

    def get_concentrations(curr_vol: float) -> Dict[str, float]:
        safe_v = max(curr_vol, 1e-9)
        return {
            "Cl_mg_l": chloride_mass / (safe_v * 1000.0),
            "SO4_mg_l": sulfate_mass / (safe_v * 1000.0),
            "Ca_mg_l": calcium_mass / (safe_v * 1000.0),
        }

    wqri_weights = config.get("wqri_weights", None)
    wqri_limits = config.get("wqri_limits", None)

    source_hardness = float(source.get("Hardness", 113.3))
    source_ph = float(source.get("pH", 7.6))

    def record_step(
        t: float,
        step: int,
        makeup_r: float = 0.0,
        blowdown_r: float = 0.0,
        makeup_on: bool = False,
        blowdown_on: bool = False,
        event_str: str = "Initial state",
        w_res: float = 0.0,
        cl_res: float = 0.0,
        so4_res: float = 0.0,
        ca_res: float = 0.0,
    ):
        concs = get_concentrations(volume)
        ec_est = calculate_ec(concs["Cl_mg_l"], source_name)
        
        # 5 First-Class Parameters:
        # 1. TDS: derived from ionic mass accumulation
        tds_est = max(ec_est * tds_factor, 1.0)
        
        # 2. Hardness: scales proportionally with calcium mass concentration
        source_ca = max(float(source["Ca"]), 1e-9)
        hardness_est = max(source_hardness * (concs["Ca_mg_l"] / source_ca), 1.0)
        
        # 3. Cycles of concentration (CoC)
        coc = max(concs["Cl_mg_l"] / max(float(source["Cl"]), 1e-9), 1.0)
        
        # 4. pH: drifts upward due to evaporative aeration & CO2 degassing
        ph_drift = 0.75 * (coc - 1.0) / (coc + 0.5)
        ph_est = float(np.clip(source_ph + ph_drift, 6.0, 9.2))
        
        # 5. Basin Water Temperature: thermal load + approach temperature
        temp_basin = float(np.clip(ambient_temp + 4.0 + 0.5 * (thermal_load / 10.0), 18.0, 42.0))
        
        # Risk Indices (LSI, SRI, CRI, WQRI)
        lsi_val = calculate_lsi(ph_est, tds_est, hardness_est, temp_c=temp_basin)
        sri_val = calculate_sri(hardness_est, ph_est, tds_est, lsi=lsi_val, limits=wqri_limits)
        cri_val = calculate_cri(ph_est, ec_est, chloride=concs["Cl_mg_l"], lsi=lsi_val, limits=wqri_limits)
        
        wqri_val = calculate_wqri(
            ph=ph_est, tds=tds_est, ec=ec_est, hardness=hardness_est,
            chloride=concs["Cl_mg_l"], sulfate=concs["SO4_mg_l"], calcium=concs["Ca_mg_l"],
            weights=wqri_weights, limits=wqri_limits
        )
        
        # Cooling Performance Derating:
        # Mineral fouling decreases heat exchanger heat transfer coefficient U
        cooling_derating_pct = float(np.clip(0.12 * sri_val * 100.0, 0.0, 15.0))
        cooling_efficiency_pct = float(100.0 - cooling_derating_pct)
        
        # Physical damage indicators
        scaling_rate_g_m2_d = float(max(0.0, 3.8 * sri_val * max(lsi_val, 0.0)))
        corrosion_rate_mpy = float(max(0.2, 2.5 * cri_val + 0.3 * (concs["Cl_mg_l"] / 4.2)))
        
        mean_age = age_inventory / max(volume, 1e-9)

        # Warning states
        ec_warning = ec_est >= ec_high
        wqri_warning = wqri_val >= wqri_trigger
        alarm_status = "CRITICAL" if (ec_warning or wqri_warning) else ("ELEVATED" if (ec_est >= ec_low or wqri_val >= 0.40) else "NORMAL")

        records.append({
            "step": step,
            "time_h": float(t),
            "basin_volume_m3": float(volume),
            "basin_fill_pct": float(100.0 * volume / basin_capacity),
            "mean_water_age_h": float(max(mean_age, 0.0)),
            "cycles_of_concentration": float(coc),
            # 5 First-Class Presentation Parameters:
            "pH": float(ph_est),
            "TDS_mg_l": float(tds_est),
            "hardness_mg_l": float(hardness_est),
            "EC_estimate_uScm": float(ec_est),
            "temperature_C": float(temp_basin),
            # Risk Indices:
            "WQRI_provisional": float(wqri_val),
            "WQRI_composite": float(wqri_val),
            "SRI_scaling_index": float(sri_val),
            "CRI_corrosion_index": float(cri_val),
            "LSI": float(lsi_val),
            # Thermodynamic Impact:
            "cooling_efficiency_pct": float(cooling_efficiency_pct),
            "scaling_rate_g_m2_d": float(scaling_rate_g_m2_d),
            "corrosion_rate_mpy": float(corrosion_rate_mpy),
            # Species & Flows:
            "Cl_mg_l": float(concs["Cl_mg_l"]),
            "SO4_mg_l": float(concs["SO4_mg_l"]),
            "Ca_mg_l": float(concs["Ca_mg_l"]),
            "makeup_m3_h": float(makeup_r),
            "blowdown_m3_h": float(blowdown_r),
            "evaporation_m3_h": float(evaporation_rate),
            "makeup_active": bool(makeup_on),
            "blowdown_active": bool(blowdown_on),
            "thermal_load_MW": float(thermal_load),
            "alarm_status": alarm_status,
            "ec_warning": bool(ec_warning),
            "wqri_warning": bool(wqri_warning),
            "switching_event": event_str,
            "water_balance_residual_m3": float(w_res),
            "chloride_balance_residual_mg": float(cl_res),
            "sulfate_balance_residual_mg": float(so4_res),
            "calcium_balance_residual_mg": float(ca_res),
        })

    # Step 0 initial snapshot
    record_step(0.0, 0)

    # Simulation loop
    for step in range(1, len(time_values)):
        t0 = float(time_values[step - 1])
        t1 = float(time_values[step])
        step_dt = t1 - t0
        if step_dt <= 0:
            continue

        prev_concs = get_concentrations(volume)
        prev_ec = calculate_ec(prev_concs["Cl_mg_l"], source_name)
        prev_tds = max(prev_ec * tds_factor, 1.0)
        source_ca = max(float(source["Ca"]), 1e-9)
        prev_hardness = max(source_hardness * (prev_concs["Ca_mg_l"] / source_ca), 1.0)
        prev_coc = max(prev_concs["Cl_mg_l"] / max(float(source["Cl"]), 1e-9), 1.0)
        prev_ph = float(np.clip(source_ph + 0.75 * (prev_coc - 1.0) / (prev_coc + 0.5), 6.0, 9.2))
        
        prev_wqri = calculate_wqri(
            ph=prev_ph, tds=prev_tds, ec=prev_ec, hardness=prev_hardness,
            chloride=prev_concs["Cl_mg_l"], sulfate=prev_concs["SO4_mg_l"], calcium=prev_concs["Ca_mg_l"],
            weights=wqri_weights, limits=wqri_limits
        )

        # ------------------------------------------------------------------
        # Controller Evaluation
        # ------------------------------------------------------------------
        makeup_active = False
        blowdown_active = False
        target_blowdown_rate = blowdown_capacity

        # Makeup logic
        if makeup_mode == "Force zero makeup":
            makeup_active = False
        elif volume < basin_capacity * 0.995:
            makeup_active = True

        # Policy-specific blowdown decision
        if policy == "Baseline":
            # Industrial Hysteresis controller:
            if prev_ec >= ec_high:
                blowdown_latched = True
            elif prev_ec <= ec_low:
                blowdown_latched = False
            blowdown_active = blowdown_latched

        elif policy == "Risk-aware":
            # Hysteresis + provisional WQRI threshold trigger
            if prev_ec >= ec_high or prev_wqri >= wqri_trigger:
                blowdown_latched = True
            elif prev_ec <= ec_low and prev_wqri < (wqri_trigger * 0.85):
                blowdown_latched = False
            blowdown_active = blowdown_latched

        elif policy == "Water-use optimization":
            # Predictive Water Conservation Controller
            ec_margin = (prev_ec - ec_low) / max(ec_high - ec_low, 1.0)
            if prev_ec >= ec_high or prev_wqri >= wqri_trigger:
                blowdown_active = True
                target_blowdown_rate = blowdown_capacity
            elif ec_margin > 0.85 or prev_wqri > (wqri_trigger * 0.90):
                blowdown_active = True
                modulation_factor = np.clip(0.3 + 0.7 * (ec_margin - 0.85) / 0.15, 0.3, 1.0)
                target_blowdown_rate = blowdown_capacity * float(modulation_factor)
            else:
                blowdown_active = False

        makeup_rate = makeup_capacity if makeup_active else 0.0
        blowdown_rate = target_blowdown_rate if blowdown_active else 0.0

        # Physical safety constraint: cannot extract more water than basin has above floor
        min_safe_volume = basin_capacity * 0.10
        max_extractable_rate = max((volume - min_safe_volume) / step_dt, 0.0)
        blowdown_rate = min(blowdown_rate, max_extractable_rate)

        # Volumetric flows in this step
        evap_vol = evaporation_rate * step_dt
        makeup_vol = makeup_rate * step_dt
        blowdown_vol = blowdown_rate * step_dt

        old_volume = volume
        old_cl_mass = chloride_mass
        old_so4_mass = sulfate_mass
        old_ca_mass = calcium_mass

        old_cl_conc = prev_concs["Cl_mg_l"]
        old_so4_conc = prev_concs["SO4_mg_l"]
        old_ca_conc = prev_concs["Ca_mg_l"]

        # Water balance update
        raw_volume = old_volume + makeup_vol - blowdown_vol - evap_vol
        new_volume = float(np.clip(raw_volume, min_safe_volume, basin_capacity))
        water_residual = (old_volume + makeup_vol - blowdown_vol - evap_vol) - new_volume

        # Species mass balances (mass in mg)
        source_cl_mass = source["Cl"] * makeup_vol * 1000.0
        source_so4_mass = source["SO4"] * makeup_vol * 1000.0
        source_ca_mass = source["Ca"] * makeup_vol * 1000.0

        blowdown_cl_mass = old_cl_conc * blowdown_vol * 1000.0
        blowdown_so4_mass = old_so4_conc * blowdown_vol * 1000.0
        blowdown_ca_mass = old_ca_conc * blowdown_vol * 1000.0

        chloride_mass = max(old_cl_mass + source_cl_mass - blowdown_cl_mass, 0.0)
        sulfate_mass = max(old_so4_mass + source_so4_mass - blowdown_so4_mass, 0.0)
        calcium_mass = max(old_ca_mass + source_ca_mass - blowdown_ca_mass, 0.0)

        cl_residual = (old_cl_mass + source_cl_mass - blowdown_cl_mass) - chloride_mass
        so4_residual = (old_so4_mass + source_so4_mass - blowdown_so4_mass) - sulfate_mass
        ca_residual = (old_ca_mass + source_ca_mass - blowdown_ca_mass) - calcium_mass

        # Water age inventory
        old_mean_age = age_inventory / max(old_volume, 1e-9)
        age_inventory += old_volume * step_dt
        age_inventory -= old_mean_age * (blowdown_vol + evap_vol)
        age_inventory = max(age_inventory, 0.0)
        age_inventory = min(age_inventory, new_volume * max(duration_h, 0.0))

        # Event tagging
        events = []
        if makeup_active:
            events.append("Makeup ON")
        if blowdown_active:
            if policy == "Water-use optimization" and target_blowdown_rate < blowdown_capacity:
                events.append(f"Blowdown MODULATED ({blowdown_rate:.1f} m³/h)")
            else:
                events.append("Blowdown ON")
        if not events:
            events.append("Circulation normal (no valve actuation)")
        if prev_ec >= ec_high:
            events.append("EC high threshold exceeded")
        elif prev_ec >= ec_low:
            events.append("EC elevated")
        if prev_wqri >= wqri_trigger:
            events.append("WQRI risk threshold triggered")

        record_step(
            t1,
            step,
            makeup_r=makeup_rate,
            blowdown_r=blowdown_rate,
            makeup_on=makeup_active,
            blowdown_on=blowdown_active,
            event_str="; ".join(events),
            w_res=water_residual,
            cl_res=cl_residual,
            so4_res=so4_residual,
            ca_res=ca_residual,
        )

        volume = new_volume

    return pd.DataFrame(records)
