# HYDROTWIN-AI — Advanced Moving 3D Digital Twin Simulator

An interactive, animated digital-twin simulator for data-center cooling-water management, built with **Python**, **Streamlit**, and **Three.js WebGL**.

---

## 🌟 Key Features

### 1. Moving 3D Digital Twin (Three.js WebGL)
- **Industrial Recirculating Basin**: Dynamic water surface level that rises and falls with simulated volume ($V/V_{cap}$), featuring wave shaders and risk-reactive coloring (Cyan $\rightarrow$ Amber $\rightarrow$ Red).
- **Data Center Heat Exchanger / Server Hall**: Server racks with dynamic thermal load dissipation (MW) and heat glow indicators.
- **Induced-Draft Cooling Towers**: High-speed axial spinning fans and dynamic vertical vapor plume particle system directly proportional to evaporation flow ($m^3/h$). Quiescent when evaporation is zero.
- **Animated Fluid Piping**: Particle streams along Catmull-Rom pipe curves with particle velocity strictly proportional to simulated flow rates. Zero flow halts particles.
- **Animated Actuator Valves**: Rotating valve stems with visual beacons (FCV-101 makeup valve, BDV-201 blowdown bleed valve).
- **Interactive OrbitControls**: Free rotation, pan, zoom, plus 1-click viewpoint presets (*Isometric*, *Top P&ID*, *Basin Zoom*, *Data Center Zoom*, *Reset Camera*).
- **Interactive Tooltip HUD**: Hover/click any 3D asset for instantaneous component telemetry.
- **60 FPS Client-Side Scrubber**: Smooth Play, Pause, Step, Reset, and Speed multiplier (0.25x to 10x) with zero lag.

### 2. 2D Engineering Process View (P&ID Fallback)
- Native HTML5 Canvas & SVG schematic operating 100% offline with zero external dependencies.
- Animated flow pulses, dynamic basin fill levels, ISO valve states, evaporation plumes, and digital sensors.
- Seamless toggling between **3D Digital Twin** and **2D Process View** without losing state.

### 3. Advanced Simulator Controls & Strategies
- **Three Primary Control Policies**:
  1. **Baseline**: Traditional EC-threshold hysteresis controller.
  2. **Risk-Aware**: Multi-objective controller evaluating electrical conductivity and provisional composite WQRI.
  3. **Water-Use Optimization**: Proactive conservation throttling that maintains high cycles of concentration near safe limits while minimizing blowdown water waste.
- **Fair Policy Comparisons**: Runs all three policies from identical starting conditions and disturbances.

### 4. Scenario Laboratory
- **Standardized Presets**:
  - *Normal Operation* (10 MW nominal, municipal freshwater)
  - *High Thermal Load* (22 MW peak computing heat wave)
  - *Poor Water Quality* (High-TDS reclaimed source)
  - *Restricted Makeup Inflow* (10 m³/h drought restriction)
  - *Restricted Blowdown Discharge* (6 m³/h sewer permit limit)
  - *Combined Multiple Stress* (High load + high TDS + restricted discharge)

### 5. Parametric Sensitivity Analysis
- Sweep thermal load, evaporation factor, EC setpoints, and pump capacities.
- Plotly curves for water consumption, conductivity risk, and limit compliance.

### 6. Diagnostics & Data Export
- Mass balance verification tracking water and dissolved species residuals.
- One-click CSV export of complete time series and comparative metrics.
- Downloadable research experiment summary report.

---

## 📐 Mathematical Formulations & Assumptions

1. **Water Mass Balance**:
   $$\frac{dV}{dt} = \dot{V}_{makeup} - \dot{V}_{blowdown} - \dot{V}_{evap}$$
   where $\dot{V}_{evap} = \text{thermal\_load} \times \text{evaporation\_factor}$.
   Basin volume is physically bounded by $0.10 \times V_{cap} \le V(t) \le V_{cap}$.

2. **Dissolved Salt Mass Conservation**:
   $$\frac{dM_i}{dt} = C_{source, i} \dot{V}_{makeup} - C_{basin, i} \dot{V}_{blowdown}$$
   for species $i \in \{\text{Cl}^-, \text{SO}_4^{2-}, \text{Ca}^{2+}\}$. Evaporation removes pure water without solute transport.

3. **Chloride-Based Conductivity Proxy**:
   $$\text{EC}_{basin} = \text{EC}_{source} \times \frac{C_{basin, \text{Cl}}}{C_{source, \text{Cl}}}$$
   *(Illustrative proxy; not an empirical multi-ion electrolyte model).*

4. **Provisional Water Quality Risk Index (WQRI)**:
   $$\text{WQRI} = \text{clip}\left( \frac{1}{3} \left[ \frac{\text{EC}}{2200} + \frac{C_{\text{Cl}}}{4.2} + \frac{C_{\text{SO}_4}}{3.4} \right], 0, 1 \right)$$
   *(Illustrative composite index; not a validated regulatory compliance score).*

---

## 🚀 Installation & Running

### Requirements
- Python 3.10+
- Dependencies: `streamlit`, `numpy`, `pandas`, `plotly`

### Launch the Application
```bash
# Activate virtual environment
.\.venv\Scripts\activate

# Run the Digital Twin simulator
streamlit run hydrotwin_twin.py
```
*(Alternatively, `streamlit run app.py` will also launch the simulator).*

### Run Automated Unit Tests
```bash
.\.venv\Scripts\python.exe -m unittest discover tests
```

---

## 📁 Repository Structure

```
hydrotwin_ai/
├── assets/
│   ├── three.min.js             # Local Three.js library for offline 3D rendering
│   └── OrbitControls.js         # Local Three.js camera OrbitControls
├── backup/
│   ├── app_orig_backup.py       # Backup of original app.py
│   └── hydrotwin_twin_orig_backup.py # Backup of original hydrotwin_twin.py
├── engine/
│   ├── models.py                # Water profiles and default configurations
│   ├── physics.py               # Mass balances, species kinetics, and simulation engine
│   ├── metrics.py               # Comparative metrics & policy benchmarking
│   ├── scenarios.py             # Scenario laboratory presets and overrides
│   └── sensitivity.py           # Parametric sensitivity sweep engine
├── visualization/
│   ├── twin_3d.py               # Standalone moving 3D WebGL Digital Twin
│   └── twin_2d.py               # Animated 2D P&ID process schematic
├── tests/
│   ├── test_simulation.py       # Unit tests for balances, zero flow, and controllers
│   └── test_visualization.py    # Unit tests for 3D/2D generation and scenarios
├── app.py                       # Application launcher
├── hydrotwin_twin.py            # Primary Streamlit Digital Twin application
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation & user guide
```
