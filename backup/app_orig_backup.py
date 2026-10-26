
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(page_title="HydroTwin-AI", layout="wide")

st.title("HydroTwin-AI")
st.caption(
    "A water-quality-aware digital-twin research prototype "
    "for data-center cooling-water management"
)


# ---------------------------------------------------------
# 1. ILLUSTRATIVE MAKEUP-WATER PROFILES
# ---------------------------------------------------------
# Example values only. Verify units and sources before research use.

SOURCES = {
    "Freshwater": {
        "ec": 345.0, "cl": 0.711, "so4": 0.332, "ca": 1.133
    },
    "HVAC condensate (illustrative)": {
        "ec": 80.0, "cl": 0.03, "so4": 0.02, "ca": 0.02
    },
    "Rainwater (illustrative)": {
        "ec": 40.0, "cl": 0.02, "so4": 0.01, "ca": 0.02
    },
    "Treated water (illustrative)": {
        "ec": 80.0, "cl": 0.02, "so4": 0.01, "ca": 0.01
    },
}


# ---------------------------------------------------------
# 2. SIMULATION INPUTS
# ---------------------------------------------------------

st.sidebar.header("Simulation inputs")

source_name = st.sidebar.selectbox(
    "Makeup-water source",
    list(SOURCES)
)
source = SOURCES[source_name]

duration = st.sidebar.slider(
    "Simulation duration (hours)",
    min_value=24,
    max_value=168,
    value=72,
    step=24,
)

dt_minutes = st.sidebar.selectbox(
    "Time step (minutes)",
    [1, 5, 15, 30],
    index=1,
)

basin_capacity = st.sidebar.number_input(
    "Basin capacity (m³)",
    min_value=10.0,
    max_value=500.0,
    value=60.0,
    step=5.0,
)

initial_volume = st.sidebar.number_input(
    "Initial basin volume (m³)",
    min_value=1.0,
    max_value=500.0,
    value=45.0,
    step=1.0,
)

if initial_volume > basin_capacity:
    st.sidebar.warning(
        "Initial volume exceeds basin capacity. "
        "The starting volume has been set to basin capacity."
    )
    initial_volume = basin_capacity

initial_water_age = st.sidebar.number_input(
    "Initial mean water age (hours)",
    min_value=0.0,
    max_value=10000.0,
    value=0.0,
    step=1.0,
    help=(
        "Set this to zero if the initial water age is unknown "
        "and you want to use a zero-age starting assumption. "
        "This assumption affects early simulation results."
    ),
)

heat_load = st.sidebar.slider(
    "Thermal load (MWth)",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1,
)

evaporation_factor = st.sidebar.number_input(
    "Evaporation factor (m³/MWhth)",
    min_value=0.1,
    max_value=3.0,
    value=1.5,
    step=0.1,
)

makeup_capacity = st.sidebar.number_input(
    "Makeup flow capacity (m³/h)",
    min_value=0.0,
    max_value=50.0,
    value=9.0,
    step=0.5,
    help="Maximum makeup flow for automatic control. Set to 0 only if you want no makeup capacity.",
)

makeup_control_mode = st.sidebar.radio(
    "Makeup control mode",
    ["Automatic controller", "Force zero makeup (test only)"],
    help=(
        "Force zero makeup disables makeup inflow regardless of basin volume. "
        "Use this option for controlled water-age tests, not normal operation."
    ),
)

blowdown_capacity = st.sidebar.number_input(
    "Blowdown flow capacity (m³/h)",
    min_value=0.1,
    max_value=50.0,
    value=4.0,
    step=0.5,
)

ec_low = st.sidebar.number_input(
    "EC lower setpoint (µS/cm)",
    min_value=0.0,
    value=1215.0,
    step=10.0,
)

ec_high = st.sidebar.number_input(
    "EC upper setpoint (µS/cm)",
    min_value=0.1,
    value=1290.0,
    step=10.0,
)

risk_trigger = st.sidebar.slider(
    "Illustrative WQRI trigger",
    min_value=0.1,
    max_value=1.0,
    value=0.50,
    step=0.05,
    help=(
        "An adjustable demonstration threshold, not an "
        "engineering limit or scientifically validated threshold."
    ),
)

policy = st.sidebar.radio(
    "Control policy",
    ["Baseline", "Risk-aware"],
)

initial_concentration_factor = st.sidebar.slider(
    "Initial concentration factor (illustrative)",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.5,
)

if ec_high <= ec_low:
    st.error("The upper EC setpoint must exceed the lower setpoint.")
    st.stop()


# ---------------------------------------------------------
# 3. MODEL ASSUMPTIONS AND INITIAL STATE
# ---------------------------------------------------------
# - The basin is assumed to be perfectly mixed.
# - Chloride, sulfate and calcium are treated as conservative species.
# - Evaporation removes water but not dissolved species.
# - Blowdown removes water and dissolved species.
# - Drift, leakage, precipitation and chemical dosing are omitted.
# - EC is a simplified chloride-based proxy.
# - WQRI is provisional and is not a validated risk model.
# - Water age is a perfectly mixed, zero-age-inflow tracer estimate.
# - Makeup enters with age zero.
# - Evaporation and blowdown remove water at the basin's mean age.
# - Initial water age is supplied by the user.
#
# Water-age equations and assumptions require research validation.
# Source concentration units must be verified before interpretation.

dt = dt_minutes / 60.0
steps = int(round(duration / dt))

ions = ["cl", "so4", "ca"]
makeup_conc = {ion: source[ion] for ion in ions}

volume = float(initial_volume)
water_age = float(initial_water_age)

concentration = {
    ion: makeup_conc[ion] * initial_concentration_factor
    for ion in ions
}

# Concentration (mmol/L) * volume (m³) * 1000 = species amount (mmol).
mass = {
    ion: concentration[ion] * volume * 1000.0
    for ion in ions
}

# Illustrative simulation safeguard, not an engineering recommendation.
minimum_volume = 0.10 * basin_capacity

evaporation_demand_rate = heat_load * evaporation_factor

makeup_on = volume < 0.85 * basin_capacity
blowdown_on = False

# Count transitions in actual interval flows, not only controller requests.
previous_makeup_active = False
previous_blowdown_active = False

rows = []


# ---------------------------------------------------------
# 4. WATER-QUALITY CALCULATION
# ---------------------------------------------------------

def calculate_quality(current_volume, current_mass):
    """Calculate concentrations, simplified EC and provisional WQRI."""

    if current_volume > 1e-9:
        conc = {
            ion: current_mass[ion] / (current_volume * 1000.0)
            for ion in ions
        }
    else:
        conc = {ion: 0.0 for ion in ions}

    if makeup_conc["cl"] > 0:
        ec = source["ec"] * conc["cl"] / makeup_conc["cl"]
    else:
        ec = source["ec"]

    ratios = [
        ec / 2200.0,
        conc["cl"] / 4.2,
        conc["so4"] / 3.4,
    ]

    wqri = float(np.clip(np.mean(ratios), 0.0, 1.0))

    return conc, float(ec), wqri


# ---------------------------------------------------------
# 5. RECORD INITIAL STATE
# ---------------------------------------------------------
# Each row represents the state at the end of its timestamp.
# Flow rates represent the interval ending at that timestamp.
# The initial row has no preceding interval.

conc, ec, wqri = calculate_quality(volume, mass)

rows.append({
    "time_h": 0.0,
    "basin_volume_m3": volume,
    "mean_water_age_h": water_age,
    "water_age_step_residual_h": 0.0,
    "EC_estimate_uScm": ec,
    "WQRI_provisional": wqri,
    "Cl_mmol_L": conc["cl"],
    "SO4_mmol_L": conc["so4"],
    "Ca_mmol_L": conc["ca"],
    "makeup_m3_h": 0.0,
    "blowdown_m3_h": 0.0,
    "evaporation_m3_h": 0.0,
    "makeup_active": False,
    "blowdown_active": False,
    "makeup_switch_event": 0,
    "blowdown_switch_event": 0,
    "water_balance_residual_m3": 0.0,
    "chloride_balance_residual_mmol": 0.0,
    "sulfate_balance_residual_mmol": 0.0,
    "calcium_balance_residual_mmol": 0.0,
})


# ---------------------------------------------------------
# 6. SIMULATION LOOP
# ---------------------------------------------------------

for step in range(1, steps + 1):

    old_volume = volume
    old_mass = mass.copy()
    old_water_age = water_age

    old_conc, ec, wqri = calculate_quality(volume, mass)

    # ---- Makeup-water controller ----
    # Turn on at or below 85%; turn off at or above 99.5%.

    if volume <= 0.85 * basin_capacity:
        makeup_on = True
    elif volume >= 0.995 * basin_capacity:
        makeup_on = False

    if makeup_control_mode == "Force zero makeup (test only)":
        desired_makeup_rate = 0.0
    else:
        desired_makeup_rate = (
            makeup_capacity if makeup_on else 0.0
        )

    # ---- Blowdown controller ----

    if policy == "Baseline":

        if ec >= ec_high:
            blowdown_on = True
        elif ec <= ec_low:
            blowdown_on = False

    else:
        risk_reset = max(0.0, risk_trigger - 0.05)

        if ec >= ec_high or wqri >= risk_trigger:
            blowdown_on = True
        elif ec <= ec_low and wqri <= risk_reset:
            blowdown_on = False

    desired_blowdown_rate = (
        blowdown_capacity if blowdown_on else 0.0
    )

    # ---- Requested interval flows ----

    desired_makeup_step = desired_makeup_rate * dt
    desired_evap_step = evaporation_demand_rate * dt
    desired_blow_step = desired_blowdown_rate * dt

    makeup_step = desired_makeup_step

    available_for_outflow = max(
        0.0,
        old_volume + makeup_step - minimum_volume
    )

    # Preserve evaporation where possible; reduce blowdown first.
    evap_step = min(
        desired_evap_step,
        available_for_outflow
    )

    remaining_after_evap = max(
        0.0,
        available_for_outflow - evap_step
    )

    blow_step = min(
        desired_blow_step,
        remaining_after_evap
    )

    # Limit inflow so that the basin does not exceed capacity.
    maximum_makeup_step = max(
        0.0,
        basin_capacity - old_volume + evap_step + blow_step
    )

    makeup_step = min(
        makeup_step,
        maximum_makeup_step
    )

    # Recheck available water after limiting makeup.
    available_total = max(
        0.0,
        old_volume + makeup_step - minimum_volume
    )

    evap_step = min(evap_step, available_total)

    blow_step = min(
        blow_step,
        max(0.0, available_total - evap_step)
    )

    # ---- Dissolved-species mass balance ----
    # Explicit step approximation: outgoing concentration is
    # approximated using the start-of-step concentration.

    for ion in ions:
        mass_in = makeup_conc[ion] * makeup_step * 1000.0
        mass_out = old_conc[ion] * blow_step * 1000.0

        mass[ion] = max(
            0.0,
            old_mass[ion] + mass_in - mass_out
        )

    # ---- Water-volume balance ----

    volume = (
        old_volume
        + makeup_step
        - evap_step
        - blow_step
    )

    water_residual = volume - (
        old_volume
        + makeup_step
        - evap_step
        - blow_step
    )

    # Record whether actual flow was active during this interval.
    makeup_active = makeup_step > 1e-12
    blowdown_active = blow_step > 1e-12
    makeup_switch_event = int(makeup_active != previous_makeup_active)
    blowdown_switch_event = int(blowdown_active != previous_blowdown_active)
    previous_makeup_active = makeup_active
    previous_blowdown_active = blowdown_active

    # ---- Mean water-age update ----
    #
    # For a perfectly mixed basin, with makeup entering at age zero
    # and all water outflows carrying the basin's mean age:
    #
    # dA/dt = 1 - (Q_makeup / V) * A
    #
    # This follows from the age-inventory balance:
    # d(V*A)/dt = V - A*(Q_blowdown + Q_evaporation).
    #
    # The differential equation is approximated using the midpoint
    # volume. Makeup flow is assumed constant over this time step.
    #
    # The exponential update below solves the equation for a
    # constant representative volume during the step.

    midpoint_volume = max(
        1e-9,
        0.5 * (old_volume + volume)
    )

    makeup_rate_actual = makeup_step / dt

    if makeup_rate_actual > 1e-12:
        exponent = makeup_step / midpoint_volume
        decay = np.exp(-exponent)

        water_age = (
            old_water_age * decay
            + (
                midpoint_volume / makeup_rate_actual
            ) * (1.0 - decay)
        )
    else:
        # With no fresh makeup, all remaining water ages by dt.
        water_age = old_water_age + dt

    water_age = float(max(0.0, water_age))

    # Residual for the trapezoidal numerical form of the age ODE.
    # This is a numerical integration diagnostic, not a validation
    # against physical water-age measurements.

    midpoint_age = 0.5 * (old_water_age + water_age)

    water_age_residual = (
        water_age
        - old_water_age
        - dt * (
            1.0
            - (
                makeup_rate_actual
                / midpoint_volume
            ) * midpoint_age
        )
    )

    # ---- End-of-step water quality ----

    conc, ec, wqri = calculate_quality(volume, mass)

    # ---- Species-balance residuals ----

    chloride_residual = mass["cl"] - (
        old_mass["cl"]
        + makeup_conc["cl"] * makeup_step * 1000.0
        - old_conc["cl"] * blow_step * 1000.0
    )

    sulfate_residual = mass["so4"] - (
        old_mass["so4"]
        + makeup_conc["so4"] * makeup_step * 1000.0
        - old_conc["so4"] * blow_step * 1000.0
    )

    calcium_residual = mass["ca"] - (
        old_mass["ca"]
        + makeup_conc["ca"] * makeup_step * 1000.0
        - old_conc["ca"] * blow_step * 1000.0
    )

    # ---- Record the end-of-step state ----

    rows.append({
        "time_h": step * dt,
        "basin_volume_m3": volume,
        "mean_water_age_h": water_age,
        "water_age_step_residual_h": water_age_residual,
        "EC_estimate_uScm": ec,
        "WQRI_provisional": wqri,
        "Cl_mmol_L": conc["cl"],
        "SO4_mmol_L": conc["so4"],
        "Ca_mmol_L": conc["ca"],
        "makeup_m3_h": makeup_step / dt,
        "blowdown_m3_h": blow_step / dt,
        "evaporation_m3_h": evap_step / dt,
        "makeup_active": makeup_active,
        "blowdown_active": blowdown_active,
        "makeup_switch_event": makeup_switch_event,
        "blowdown_switch_event": blowdown_switch_event,
        "water_balance_residual_m3": water_residual,
        "chloride_balance_residual_mmol": chloride_residual,
        "sulfate_balance_residual_mmol": sulfate_residual,
        "calcium_balance_residual_mmol": calcium_residual,
    })


# ---------------------------------------------------------
# 7. RESULTS AND CUMULATIVE FLOWS
# ---------------------------------------------------------

df = pd.DataFrame(rows)

# Row zero has no interval flow. Each later row records flows
# during the interval ending at that row.

df["cumulative_makeup_m3"] = (
    df["makeup_m3_h"] * dt
).cumsum()

df["cumulative_blowdown_m3"] = (
    df["blowdown_m3_h"] * dt
).cumsum()

df["cumulative_evaporation_m3"] = (
    df["evaporation_m3_h"] * dt
).cumsum()

last = df.iloc[-1]

total_makeup = float(df["cumulative_makeup_m3"].iloc[-1])
total_blowdown = float(df["cumulative_blowdown_m3"].iloc[-1])
total_evaporation = float(df["cumulative_evaporation_m3"].iloc[-1])
makeup_switch_count = int(df["makeup_switch_event"].sum())
blowdown_switch_count = int(df["blowdown_switch_event"].sum())

last_24h_start = max(0.0, float(duration) - 24.0)
last_24h_df = df[df["time_h"] >= last_24h_start]
last_24h_average_age = float(last_24h_df["mean_water_age_h"].mean())
minimum_volume_reached = bool(
    (df["basin_volume_m3"] <= minimum_volume + 1e-6).any()
)
requested_evaporation_total = evaporation_demand_rate * duration
evaporation_shortfall = max(0.0, requested_evaporation_total - total_evaporation)

max_water_residual = float(
    df["water_balance_residual_m3"].abs().max()
)

max_chloride_residual = float(
    df["chloride_balance_residual_mmol"].abs().max()
)

max_sulfate_residual = float(
    df["sulfate_balance_residual_mmol"].abs().max()
)

max_calcium_residual = float(
    df["calcium_balance_residual_mmol"].abs().max()
)

max_age_residual = float(
    df["water_age_step_residual_h"].abs().max()
)

minimum_water_age = float(
    df["mean_water_age_h"].min()
)

maximum_water_age = float(
    df["mean_water_age_h"].max()
)


# ---------------------------------------------------------
# 8. DASHBOARD SUMMARY
# ---------------------------------------------------------

st.subheader("Simulation summary")

a, b, c, d, e, f, g = st.columns(7)

a.metric(
    "Final basin volume",
    f"{last['basin_volume_m3']:.2f} m³"
)

b.metric(
    "Final estimated EC",
    f"{last['EC_estimate_uScm']:.0f} µS/cm"
)

c.metric(
    "Total makeup",
    f"{total_makeup:.2f} m³"
)

d.metric(
    "Total blowdown",
    f"{total_blowdown:.2f} m³"
)

e.metric(
    "Final provisional WQRI",
    f"{last['WQRI_provisional']:.2f}"
)

f.metric(
    "Final mean water age",
    f"{last['mean_water_age_h']:.2f} h"
)

g.metric(
    "Last 24 h average age",
    f"{last_24h_average_age:.2f} h"
)

st.caption(
    f"Policy: {policy} | Makeup mode: {makeup_control_mode} | "
    f"Source: {source_name} | Duration: {duration} h | "
    f"Time step: {dt_minutes} min"
)

# ---------------------------------------------------------
# 8A. MODEL-CONNECTED VIRTUAL COOLING-WATER PLANT
# ---------------------------------------------------------

st.subheader("Virtual Cooling-Water Plant")
st.caption(
    "Plant indicators use the simulated final state. "
    "The recirculation loop is schematic because recirculation "
    "flow and heat-exchanger dynamics are not explicitly modeled."
)

# Read the final simulation state
twin_volume = float(last["basin_volume_m3"])
twin_level = 100.0 * twin_volume / basin_capacity

twin_ec = float(last["EC_estimate_uScm"])
twin_wqri = float(last["WQRI_provisional"])
twin_age = float(last["mean_water_age_h"])

twin_makeup = float(last["makeup_m3_h"])
twin_blowdown = float(last["blowdown_m3_h"])
twin_evaporation = float(last["evaporation_m3_h"])

twin_makeup_on = bool(last["makeup_active"])
twin_blowdown_on = bool(last["blowdown_active"])

twin_basin_status = (
    "LOW LEVEL" if twin_volume <= minimum_volume + 1e-6
    else "NORMAL"
)

twin_ec_status = (
    "HIGH EC" if twin_ec >= ec_high
    else "CONTROL BAND" if twin_ec >= ec_low
    else "BELOW UPPER BAND"
)

# SVG basin dimensions
tank_y = 270.0
tank_height = 120.0
water_height = tank_height * min(
    1.0, max(0.0, twin_volume / basin_capacity)
)
water_y = tank_y + tank_height - water_height

# Flow animations represent schematic direction only.
# Makeup/blowdown animations reflect the actual final interval.
makeup_class = "" if twin_makeup_on else "stopped"
blowdown_class = "" if twin_blowdown_on else "stopped"

plant_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* { box-sizing: border-box; }

body {
    margin: 0;
    padding: 8px;
    background: #f1f6fa;
    color: #18344d;
    font-family: Arial, sans-serif;
}

.panel {
    background: #fff;
    border: 1px solid #d8e5ee;
    border-radius: 16px;
    padding: 12px;
}

.heading {
    font-size: 19px;
    font-weight: 700;
    margin: 4px 8px;
}

.subheading {
    font-size: 12px;
    color: #668094;
    margin: 0 8px 8px;
}

svg {
    width: 100%;
    height: auto;
    display: block;
}

.pipe {
    fill: none;
    stroke: #8398a8;
    stroke-width: 11;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.flow {
    fill: none;
    stroke: #1da8d5;
    stroke-width: 4;
    stroke-dasharray: 10 12;
    stroke-linecap: round;
    animation: flow 1.2s linear infinite;
}

.slow { animation-duration: 2.2s; }

.stopped {
    animation: none;
    stroke: #9eacb6;
    opacity: 0.5;
}

@keyframes flow {
    to { stroke-dashoffset: -44; }
}

.equipment {
    fill: #e9f2f7;
    stroke: #7190a5;
    stroke-width: 2.5;
}

.label {
    fill: #193c56;
    font-size: 14px;
    font-weight: bold;
}

.small {
    fill: #617b8d;
    font-size: 11px;
}

.value {
    fill: #087eae;
    font-size: 13px;
    font-weight: bold;
}

.water {
    fill: #42b9e5;
    opacity: 0.85;
}

.sensor {
    fill: #edf9ff;
    stroke: #a4d6e9;
    stroke-width: 1.5;
}

.status {
    fill: #e6f7ed;
    stroke: #80cba0;
    stroke-width: 1.5;
}

.warning {
    fill: #fff3dc;
    stroke: #e7b34e;
    stroke-width: 1.5;
}

.off {
    fill: #eef1f4;
    stroke: #a8b4bd;
    stroke-width: 1.5;
}

</style>
</head>
<body>
<div class="panel">
    <div class="heading">HYDROTWIN-AI | Process Overview</div>
    <div class="subheading">
        Simulated basin state, water-quality indicators and controller flows
    </div>

    <svg viewBox="0 0 1000 510"
         role="img"
         aria-label="Virtual cooling-water system">

        <defs>
            <clipPath id="tankClip">
                <rect x="280" y="270" width="330" height="120" rx="5"/>
            </clipPath>
            <marker id="arrow" markerWidth="8" markerHeight="8"
                    refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#1da8d5"/>
            </marker>
        </defs>

        <!-- Schematic recirculation: basin -> pump -> heat exchanger
             -> cooling tower -> basin. Not a calculated flow rate. -->

        <path class="pipe"
              d="M610 320 H685"/>
        <path class="pipe"
              d="M715 320 H925 V150 H875"/>
        <path class="pipe"
              d="M760 110 H520"/>
        <path class="pipe"
              d="M435 155 V270"/>

        <path class="flow slow"
              d="M610 320 H685" marker-end="url(#arrow)"/>
        <path class="flow slow"
              d="M715 320 H925 V150 H875" marker-end="url(#arrow)"/>
        <path class="flow slow"
              d="M760 110 H520" marker-end="url(#arrow)"/>
        <path class="flow slow"
              d="M435 155 V270" marker-end="url(#arrow)"/>

        <!-- Cooling tower -->
        <path class="equipment"
              d="M365 48 L390 38 H480 L505 48
                 L490 150 H380 Z"/>
        <path d="M380 78 H495 M383 98 H492 M386 118 H489"
              fill="none" stroke="#9bb1bf" stroke-width="3"/>
        <path d="M400 38 V25 H470 V38"
              fill="#e9f2f7" stroke="#7190a5" stroke-width="2"/>

        <text x="435" y="175" text-anchor="middle" class="label">
            COOLING TOWER
        </text>
        <text x="435" y="192" text-anchor="middle" class="small">
            Heat rejection schematic
        </text>

        <!-- Basin -->
        <rect x="270" y="260" width="350" height="140" rx="9"
              fill="#dce8ef" stroke="#7190a5" stroke-width="3"/>

        <g clip-path="url(#tankClip)">
            <rect x="280" y="{{WATER_Y}}"
                  width="330" height="{{WATER_HEIGHT}}"
                  class="water"/>
            <path d="M280 300 Q300 294 320 300 T360 300 T400 300
                     T440 300 T480 300 T520 300 T560 300 T610 300"
                  fill="none" stroke="#c6f1ff" stroke-width="2"/>
        </g>

        <text x="445" y="285" text-anchor="middle" class="label">
            RECIRCULATING BASIN
        </text>
        <text x="445" y="315" text-anchor="middle" class="value">
            {{VOLUME}} / {{CAPACITY}} m³
        </text>
        <text x="445" y="335" text-anchor="middle" class="small">
            Level: {{LEVEL}}%
        </text>
        <text x="445" y="355" text-anchor="middle" class="small">
            Mean age: {{AGE}} h
        </text>

        <!-- Pump -->
        <circle cx="700" cy="320" r="28"
                fill="#e4f6fc" stroke="#398eb4" stroke-width="3"/>
        <path d="M688 304 L714 320 L688 336 Z"
              fill="#398eb4"/>
        <text x="700" y="365" text-anchor="middle" class="label">PUMP</text>
        <text x="700" y="382" text-anchor="middle" class="small">
            Loop schematic
        </text>

        <!-- Heat exchanger / data-centre load -->
        <rect x="760" y="65" width="115" height="85" rx="9"
              class="equipment"/>
        <path d="M778 83 V130 M793 83 V130 M808 83 V130
                 M823 83 V130 M838 83 V130"
              stroke="#80a2b7" stroke-width="3"/>
        <text x="817" y="48" text-anchor="middle" class="label">
            HEAT EXCHANGER
        </text>
        <text x="817" y="166" text-anchor="middle" class="value">
            {{LOAD}} MWth
        </text>

        <!-- Makeup inlet: actual final interval flow -->
        <path class="pipe" d="M55 320 H270"/>
        <path class="flow {{MAKEUP_CLASS}}"
              d="M55 320 H270" marker-end="url(#arrow)"/>
        <text x="55" y="283" class="label">MAKEUP WATER</text>
        <text x="55" y="302" class="value">{{MAKEUP}} m³/h</text>
        <text x="55" y="345" class="small">Controller: {{MAKEUP_STATE}}</text>

        <!-- Blowdown outlet: actual final interval flow -->
        <path class="pipe" d="M445 400 V450 H650"/>
        <path class="flow {{BLOWDOWN_CLASS}}"
              d="M445 400 V450 H650" marker-end="url(#arrow)"/>
        <text x="475" y="430" class="label">BLOWDOWN</text>
        <text x="475" y="470" class="value">{{BLOWDOWN}} m³/h</text>
        <text x="650" y="490" class="small">Discharge</text>

        <!-- Water-quality sensor panel -->
        <rect x="40" y="65" width="235" height="145" rx="10"
              class="sensor"/>
        <text x="58" y="91" class="label">WATER QUALITY</text>
        <text x="58" y="119" class="value">EC: {{EC}} µS/cm</text>
        <text x="58" y="143" class="value">WQRI: {{WQRI}}</text>
        <text x="58" y="167" class="value">Cl: {{CL}} mmol/L</text>
        <text x="58" y="189" class="value">SO₄: {{SO4}} mmol/L</text>

        <!-- Evaporation indicator -->
        <rect x="720" y="220" width="235" height="75" rx="10"
              class="sensor"/>
        <text x="738" y="245" class="label">EVAPORATION</text>
        <text x="738" y="269" class="value">{{EVAP}} m³/h</text>
        <text x="738" y="285" class="small">
            Model-calculated water loss
        </text>

        <!-- Status panel -->
        <rect x="720" y="405" width="235" height="65" rx="10"
              class="{{STATUS_CLASS}}"/>
        <text x="738" y="429" class="label">BASIN: {{STATUS}}</text>
        <text x="738" y="450" class="small">EC: {{EC_STATUS}}</text>

        <text x="985" y="505" text-anchor="end" class="small">
            Simulation-based prototype
        </text>
    </svg>
</div>
</body>
</html>
"""

# Replace all model-linked display values
replacements = {
    "{{WATER_Y}}": f"{water_y:.2f}",
    "{{WATER_HEIGHT}}": f"{water_height:.2f}",
    "{{VOLUME}}": f"{twin_volume:.2f}",
    "{{CAPACITY}}": f"{basin_capacity:.1f}",
    "{{LEVEL}}": f"{twin_level:.1f}",
    "{{AGE}}": f"{twin_age:.2f}",
    "{{LOAD}}": f"{heat_load:.1f}",
    "{{MAKEUP}}": f"{twin_makeup:.2f}",
    "{{BLOWDOWN}}": f"{twin_blowdown:.2f}",
    "{{EVAP}}": f"{twin_evaporation:.2f}",
    "{{EC}}": f"{twin_ec:.1f}",
    "{{WQRI}}": f"{twin_wqri:.3f}",
    "{{CL}}": f"{float(last['Cl_mmol_L']):.3f}",
    "{{SO4}}": f"{float(last['SO4_mmol_L']):.3f}",
    "{{MAKEUP_CLASS}}": makeup_class,
    "{{BLOWDOWN_CLASS}}": blowdown_class,
    "{{MAKEUP_STATE}}": "ON" if twin_makeup_on else "OFF",
    "{{STATUS}}": twin_basin_status,
    "{{EC_STATUS}}": twin_ec_status,
    "{{STATUS_CLASS}}": (
        "warning" if twin_volume <= minimum_volume + 1e-6
        or twin_ec >= ec_high else "status"
    ),
}

for key, value in replacements.items():
    plant_html = plant_html.replace(key, value)

components.html(
    plant_html,
    height=620,
    scrolling=False,
)

# Model-linked ion and controller summary
p1, p2, p3, p4, p5 = st.columns(5)

p1.metric("Chloride", f"{last['Cl_mmol_L']:.3f} mmol/L")
p2.metric("Sulfate", f"{last['SO4_mmol_L']:.3f} mmol/L")
p3.metric("Calcium", f"{last['Ca_mmol_L']:.3f} mmol/L")
p4.metric("Makeup controller", "ON" if twin_makeup_on else "OFF")
p5.metric("Blowdown controller", "ON" if twin_blowdown_on else "OFF")

st.caption(
    "EC and WQRI are model estimates. The equipment drawing is schematic; "
    "the current model does not calculate pump speed, recirculation flow, "
    "heat-exchanger temperatures, or detailed cooling-tower physics."
)


if minimum_volume_reached:
    st.warning(
        f"Minimum basin volume safeguard reached ({minimum_volume:.2f} m³). "
        "The simulator curtailed water outflow to preserve this floor. "
        "If evaporation demand was curtailed, the requested cooling-water "
        "evaporation demand was not met in this simulation. The prototype "
        "does not model thermal trips or verify heat rejection."
    )

if evaporation_shortfall > 1e-6:
    st.warning(
        f"Evaporation demand shortfall: {evaporation_shortfall:.2f} m³ "
        f"(requested {requested_evaporation_total:.2f} m³; "
        f"simulated {total_evaporation:.2f} m³)."
    )


# ---------------------------------------------------------
# 9. CHARTS AND DATA
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Water system",
    "Water quality",
    "Simulation data",
    "Model diagnostics",
])


with tab1:

    fig_volume = px.line(
        df,
        x="time_h",
        y="basin_volume_m3",
        title="Basin water volume over time",
        labels={
            "time_h": "Simulation time (hours)",
            "basin_volume_m3": "Water volume (m³)",
        },
    )

    fig_volume.add_hline(
        y=minimum_volume,
        line_dash="dash",
        annotation_text="Illustrative minimum volume",
    )

    fig_volume.add_hline(
        y=basin_capacity,
        line_dash="dash",
        annotation_text="Basin capacity",
    )

    st.plotly_chart(fig_volume, use_container_width=True)

    # ---- Water-age chart ----

    st.subheader("Water-age tracking")

    fig_age = px.line(
        df,
        x="time_h",
        y="mean_water_age_h",
        title="Estimated mean water age over time",
        labels={
            "time_h": "Simulation time (hours)",
            "mean_water_age_h": "Mean water age (hours)",
        },
    )

    st.plotly_chart(fig_age, use_container_width=True)

    st.caption(
        "Estimated mean age of water in the basin under the "
        "perfect-mixing and zero-age-makeup assumptions. "
        "This is a simulated state, not a sensor measurement."
    )

    flow_df = df[
        [
            "time_h",
            "makeup_m3_h",
            "blowdown_m3_h",
            "evaporation_m3_h",
        ]
    ].melt(
        id_vars="time_h",
        var_name="Flow type",
        value_name="Flow rate (m³/h)",
    )

    flow_names = {
        "makeup_m3_h": "Makeup",
        "blowdown_m3_h": "Blowdown",
        "evaporation_m3_h": "Evaporation",
    }

    flow_df["Flow type"] = flow_df["Flow type"].map(flow_names)

    fig_flows = px.line(
        flow_df,
        x="time_h",
        y="Flow rate (m³/h)",
        color="Flow type",
        title="Water flow rates",
        labels={"time_h": "Simulation time (hours)"},
    )

    st.plotly_chart(fig_flows, use_container_width=True)

    st.subheader("Cumulative water use")

    cumulative_df = df[
        [
            "time_h",
            "cumulative_makeup_m3",
            "cumulative_blowdown_m3",
            "cumulative_evaporation_m3",
        ]
    ].melt(
        id_vars="time_h",
        var_name="Flow type",
        value_name="Cumulative volume (m³)",
    )

    cumulative_names = {
        "cumulative_makeup_m3": "Makeup",
        "cumulative_blowdown_m3": "Blowdown",
        "cumulative_evaporation_m3": "Evaporation",
    }

    cumulative_df["Flow type"] = cumulative_df["Flow type"].map(
        cumulative_names
    )

    fig_cumulative = px.line(
        cumulative_df,
        x="time_h",
        y="Cumulative volume (m³)",
        color="Flow type",
        title="Cumulative water flows",
        labels={"time_h": "Simulation time (hours)"},
    )

    st.plotly_chart(fig_cumulative, use_container_width=True)


with tab2:

    fig_ec = px.line(
        df,
        x="time_h",
        y="EC_estimate_uScm",
        title="Estimated electrical conductivity",
        labels={
            "time_h": "Simulation time (hours)",
            "EC_estimate_uScm": "Estimated EC (µS/cm)",
        },
    )

    fig_ec.add_hline(
        y=ec_low,
        line_dash="dash",
        annotation_text="EC lower setpoint",
    )

    fig_ec.add_hline(
        y=ec_high,
        line_dash="dash",
        annotation_text="EC upper setpoint",
    )

    st.plotly_chart(fig_ec, use_container_width=True)

    fig_wqri = px.line(
        df,
        x="time_h",
        y="WQRI_provisional",
        title="Provisional water-quality risk indicator",
        labels={
            "time_h": "Simulation time (hours)",
            "WQRI_provisional": "Provisional WQRI (0–1)",
        },
    )

    fig_wqri.add_hline(
        y=risk_trigger,
        line_dash="dash",
        annotation_text="Illustrative WQRI trigger",
    )

    st.plotly_chart(fig_wqri, use_container_width=True)

    concentration_df = df[
        ["time_h", "Cl_mmol_L", "SO4_mmol_L", "Ca_mmol_L"]
    ].melt(
        id_vars="time_h",
        var_name="Species",
        value_name="Concentration (mmol/L)",
    )

    species_names = {
        "Cl_mmol_L": "Chloride",
        "SO4_mmol_L": "Sulfate",
        "Ca_mmol_L": "Calcium",
    }

    concentration_df["Species"] = concentration_df["Species"].map(
        species_names
    )

    fig_species = px.line(
        concentration_df,
        x="time_h",
        y="Concentration (mmol/L)",
        color="Species",
        title="Dissolved-species concentrations",
        labels={"time_h": "Simulation time (hours)"},
    )

    st.plotly_chart(fig_species, use_container_width=True)

    st.info(
        "EC is estimated from chloride concentration, and WQRI is an "
        "illustrative indicator based on EC, chloride and sulfate. "
        "These are not independent, validated water-quality measurements."
    )


with tab3:

    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download simulation CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hydrotwin_simulation.csv",
        mime="text/csv",
    )


with tab4:

    st.subheader("Water and species mass-balance checks")

    diagnostic_df = pd.DataFrame({
        "Diagnostic": [
            "Maximum absolute water-balance residual (m³)",
            "Maximum absolute chloride-balance residual (mmol)",
            "Maximum absolute sulfate-balance residual (mmol)",
            "Maximum absolute calcium-balance residual (mmol)",
        ],
        "Value": [
            max_water_residual,
            max_chloride_residual,
            max_sulfate_residual,
            max_calcium_residual,
        ],
    })

    st.dataframe(diagnostic_df, use_container_width=True)

    st.write(
        f"Minimum simulated volume: "
        f"{df['basin_volume_m3'].min():.4f} m³"
    )

    st.write(
        f"Maximum simulated volume: "
        f"{df['basin_volume_m3'].max():.4f} m³"
    )

    st.write(f"Total makeup: {total_makeup:.2f} m³")
    st.write(f"Total blowdown: {total_blowdown:.2f} m³")
    st.write(f"Total evaporation: {total_evaporation:.2f} m³")
    st.write(f"Requested evaporation demand: {requested_evaporation_total:.2f} m³")
    st.write(f"Evaporation demand shortfall: {evaporation_shortfall:.2f} m³")
    st.write(f"Makeup on/off transitions (actual flow): {makeup_switch_count}")
    st.write(f"Blowdown on/off transitions (actual flow): {blowdown_switch_count}")
    st.write(f"Average water age over final 24 h: {last_24h_average_age:.4f} h")

    if minimum_volume_reached:
        st.warning(
            f"Minimum volume floor reached: {minimum_volume:.2f} m³. "
            "This is a numerical safeguard, not a validated operating limit."
        )
    else:
        st.success("Minimum volume floor was not reached in this run.")

    if (
        max_water_residual < 1e-8
        and max_chloride_residual < 1e-6
        and max_sulfate_residual < 1e-6
        and max_calcium_residual < 1e-6
    ):
        st.success(
            "The implemented water and dissolved-species balance "
            "equations pass their numerical residual checks."
        )
    else:
        st.error(
            "One or more balance residuals are larger than expected. "
            "Review the model calculations before interpreting results."
        )

    st.caption(
        "Passing these checks means the implemented equations are "
        "internally consistent. It does not validate the model against "
        "a real cooling-water system."
    )

    # ---- Water-age diagnostics ----

    st.subheader("Water-age diagnostics")

    age_diagnostic_df = pd.DataFrame({
        "Diagnostic": [
            "Initial mean water age (hours)",
            "Final mean water age (hours)",
            "Minimum mean water age (hours)",
            "Maximum mean water age (hours)",
            "Average water age over final 24 h (hours)",
            "Total makeup (m³)",
            "Total blowdown (m³)",
            "Makeup on/off transitions (actual flow)",
            "Blowdown on/off transitions (actual flow)",
            "Maximum absolute age integration residual (hours)",
        ],
        "Value": [
            initial_water_age,
            float(last["mean_water_age_h"]),
            minimum_water_age,
            maximum_water_age,
            last_24h_average_age,
            total_makeup,
            total_blowdown,
            makeup_switch_count,
            blowdown_switch_count,
            max_age_residual,
        ],
    })

    # Render this table as HTML so both diagnostic labels and values can
    # use a larger, more readable font than Streamlit's default dataframe.
    st.markdown(
        """
        <style>
        table.water-age-diagnostics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 19px;
        }
        table.water-age-diagnostics-table th {
            font-size: 20px;
            font-weight: 700;
            text-align: left;
            padding: 12px 14px;
            border-bottom: 2px solid rgba(128, 128, 128, 0.45);
        }
        table.water-age-diagnostics-table td {
            font-size: 19px;
            padding: 11px 14px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        age_diagnostic_df.to_html(
            index=False,
            border=0,
            classes="water-age-diagnostics-table",
            escape=True,
        ),
        unsafe_allow_html=True,
    )

    if (
        np.isfinite(df["mean_water_age_h"]).all()
        and (df["mean_water_age_h"] >= 0).all()
    ):
        st.success(
            "Water-age sanity checks passed: all simulated values "
            "are finite and non-negative."
        )
    else:
        st.error(
            "Water-age sanity checks failed. Review the age calculation."
        )

    st.caption(
        "The age integration residual checks numerical consistency "
        "with the discretized age equation. It does not establish "
        "physical accuracy or validate the age model."
    )


# ---------------------------------------------------------
# 10. RESEARCH LIMITATIONS
# ---------------------------------------------------------

st.warning(
    "Research prototype only. The force-zero-makeup option is a test-only "
    "override. The source-water profiles, evaporation "
    "factor, EC proxy, WQRI formula and thresholds are illustrative and "
    "not validated against a real data center. Water age is estimated "
    "using a perfectly mixed basin model, a user-specified initial age, "
    "zero-age makeup water, and the assumption that evaporative and "
    "blowdown outflows carry the basin's mean age. These assumptions "
    "require literature justification and validation. The model omits "
    "drift, leakage, pH, alkalinity, scaling chemistry, precipitation, "
    "corrosion and detailed cooling-tower physics. The minimum basin "
    "volume is a simulation safeguard, not an engineering recommendation. "
    "Do not interpret the outputs as measured facility performance or "
    "validated predictions."
)
