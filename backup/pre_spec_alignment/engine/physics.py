"""
HYDROTWIN-AI: Physical Simulation Engine
Implements mass balances, dissolved species kinetics, EC proxy estimation,
and provisional WQRI tracking for recirculating cooling water.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from engine.models import WATER_PROFILES


def calculate_ec(chloride_mg_l: float, source_name: str) -> float:
    """
    Illustrative chloride-based electrical conductivity (EC) proxy.
    
    Disclaimer:
    This is an illustrative linear proxy, not an empirical multi-ion conductivity model.
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


def calculate_wqri(
    ec: float,
    chloride: float,
    sulfate: float,
    calcium: float = 0.0,
    weights: Dict[str, float] = None,
    limits: Dict[str, float] = None,
) -> float:
    """
    Transparent, Configurable Water Quality Risk Index (WQRI).
    
    Combines normalized key water-chemistry risk proxies:
    - Electrical Conductivity (EC / EC_max): general dissolved solids & ionic activity
    - Chloride (Cl / Cl_max): localized pitting corrosion risk for stainless steels & heat exchangers
    - Sulfate (SO4 / SO4_max): acidic corrosion and calcium sulfate scaling potential
    - Calcium (Ca / Ca_max): mineral scaling and heat-transfer foulant risk
    
    Equation:
      WQRI = clip( sum(w_i * (C_i / C_i,ref)), 0.0, 1.0 )
    
    Disclaimer:
    This index is an engineering composite risk proxy for digital-twin decision-making.
    It is not an officially promulgated regulatory compliance index.
    """
    ref_limits = {
        "ec": 2200.0,      # µS/cm upper industrial cooling threshold
        "cl": 4.2,         # mg/L scaled reference
        "so4": 3.4,        # mg/L scaled reference
        "ca": 5.0,         # mg/L scaled reference
    }
    if limits:
        ref_limits.update(limits)
        
    c_ec = max(float(ec), 0.0) / max(float(ref_limits["ec"]), 1e-6)
    c_cl = max(float(chloride), 0.0) / max(float(ref_limits["cl"]), 1e-6)
    c_so4 = max(float(sulfate), 0.0) / max(float(ref_limits["so4"]), 1e-6)
    c_ca = max(float(calcium), 0.0) / max(float(ref_limits["ca"]), 1e-6)
    
    if weights is not None:
        w_ec = float(weights.get("ec", 0.35))
        w_cl = float(weights.get("cl", 0.30))
        w_so4 = float(weights.get("so4", 0.20))
        w_ca = float(weights.get("ca", 0.15))
        w_sum = max(w_ec + w_cl + w_so4 + w_ca, 1e-6)
        wqri_val = (w_ec * c_ec + w_cl * c_cl + w_so4 * c_so4 + w_ca * c_ca) / w_sum
    else:
        # Backward-compatible 3-component formulation when calcium is omitted
        if calcium <= 0.0:
            wqri_val = (c_ec + c_cl + c_so4) / 3.0
        else:
            # 4-component balanced default
            wqri_val = 0.35 * c_ec + 0.30 * c_cl + 0.20 * c_so4 + 0.15 * c_ca
            
    return float(np.clip(wqri_val, 0.0, 1.0))



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
    Simulates a well-mixed recirculating cooling basin using mass balances.
    
    Mechanisms:
    1. Evaporation removes pure water (zero dissolved salt loss), concentrating ions.
    2. Makeup adds water and fresh dissolved species from the source profile.
    3. Blowdown discharges mixed basin water and dissolved species.
    4. Controller determines makeup and blowdown valve actuation based on policy.
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

    n_steps = int(np.ceil(duration_h / dt_h))
    time_values = np.minimum(
        np.arange(n_steps + 1, dtype=float) * dt_h,
        duration_h,
    )

    # Constant or variable evaporation rate (m³/h)
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
        wqri_val = calculate_wqri(
            ec_est, concs["Cl_mg_l"], concs["SO4_mg_l"], concs["Ca_mg_l"],
            weights=wqri_weights, limits=wqri_limits
        )
        mean_age = age_inventory / max(volume, 1e-9)

        # Cycles of concentration (CoC) relative to makeup source
        source_cl = max(float(source["Cl"]), 1e-9)
        coc = concs["Cl_mg_l"] / source_cl

        # Warning states
        ec_warning = ec_est >= ec_high
        wqri_warning = wqri_val >= wqri_trigger
        alarm_status = "CRITICAL" if (ec_warning or wqri_warning) else ("ELEVATED" if ec_est >= ec_low else "NORMAL")

        records.append({
            "step": step,
            "time_h": float(t),
            "basin_volume_m3": float(volume),
            "basin_fill_pct": float(100.0 * volume / basin_capacity),
            "mean_water_age_h": float(max(mean_age, 0.0)),
            "cycles_of_concentration": float(coc),
            "EC_estimate_uScm": float(ec_est),
            "WQRI_provisional": float(wqri_val),
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
        prev_wqri = calculate_wqri(
            prev_ec, prev_concs["Cl_mg_l"], prev_concs["SO4_mg_l"], prev_concs["Ca_mg_l"],
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
            # Turn ON when EC >= ec_high; Turn OFF when EC <= ec_low
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
            # Predictive Water Conservation Controller:
            # Operates closer to upper threshold without violation by modulating blowdown rate.
            # When EC is well below threshold, zero blowdown.
            # When EC enters high band (>= 92% of ec_high) or WQRI >= 90% of trigger,
            # apply proportionate proportional-integral throttling to maintain equilibrium.
            ec_margin = (prev_ec - ec_low) / max(ec_high - ec_low, 1.0)
            if prev_ec >= ec_high or prev_wqri >= wqri_trigger:
                blowdown_active = True
                target_blowdown_rate = blowdown_capacity
            elif ec_margin > 0.85 or prev_wqri > (wqri_trigger * 0.90):
                blowdown_active = True
                # Proportional modulation: throttle flow between 30% and 80% of capacity
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
