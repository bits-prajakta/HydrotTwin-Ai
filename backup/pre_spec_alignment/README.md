# HYDROTWIN-AI — Water-Quality-Aware Digital Twin for Data-Center Cooling Management

An advanced, presentation-ready scientific digital-twin research prototype combining **physical mass-balance simulation**, **predictive machine learning (Random Forest, Gradient Boosting)**, **explainable AI (XAI)**, **multi-objective water conservation control**, and an **interactive moving 3D/2D digital twin**.

---

## 🌟 Key Features & Research Modules

### 1. Dual-Engine Moving Digital Twin (3D WebGL & 2D Canvas)
- **3D Industrial Twin (Three.js WebGL)**:
  - Recirculating cooling-water basin with dynamic water level, wave shaders, and risk-reactive coloring.
  - Server hall thermal load dissipation section with dynamic heat flux indicators.
  - Induced-draft cooling towers with high-speed spinning fans and dynamic vertical vapor plume particle systems directly linked to thermal evaporation flow ($m^3/h$).
  - Animated fluid pipelines with particle streams whose velocity is strictly proportional to simulated flow rates. Zero flow halts particles.
  - Actuator valves with visual status beacons (FCV-101 makeup valve, BDV-201 blowdown bleed valve).
  - OrbitControls: Orbit, zoom, pan, camera reset, and 1-click viewpoint presets.
  - 60 FPS client-side playback console with Play, Pause, Step, Reset, and Speed multipliers.
- **2D Engineering Process View (P&ID Schematic Fallback)**:
  - Native HTML5 Canvas schematic operating 100% offline with zero external dependencies.
  - Flow pulses, dynamic tank levels, ISO valve states, digital sensor readouts, and synced scrubber.
  - Seamless switching between 3D and 2D views with zero state loss.

### 2. Physical & Chemical Simulation Engine
- **Basin Water-Volume Balance**: Continuous mass conservation accounting for makeup inflow, blowdown bleed, and thermal evaporation.
- **Thermodynamic Evaporative Coupling**: Explicitly derived from water latent heat of vaporization ($\alpha \approx 1.47\text{–}1.60 \text{ m}^3/(\text{MW}\cdot\text{h})$).
- **Individual Dissolved-Species Mass Balances**: Conservation of non-volatile salts (Chloride, Sulfate, Calcium). Evaporation removes pure vapor; makeup adds salts; blowdown purges salts.
- **Conductivity (EC) & Composite Risk Index (WQRI)**:
  - Proportional chloride-proxy electrical conductivity model.
  - Multi-objective Water Quality Risk Index (WQRI) incorporating pitting, scaling, and general ionic activity with configurable weights and reference limits.
- **Zero Numerical Residual Drift**: Water and dissolved-species residuals verified at $< 10^{-10} \text{ mg}$.

### 3. Predictive Machine Learning & Explainable AI (XAI)
- **Supervised Forecasting Tasks**:
  1. Multi-step forward Electrical Conductivity ($\text{EC}_{t+k}$) forecasting over configurable horizons (1.0 to 4.0 hours).
  2. Water-quality risk category classification (Normal, Elevated, Critical).
- **Rigorous Model Benchmarking**:
  - Compares Naive Persistence Baseline, Ridge Regression, Random Forest, and Gradient Boosting.
  - Compares Majority Class Baseline, Logistic Regression, and Random Forest Classifier.
  - Evaluates MAE, RMSE, and $R^2$ for regression; Precision, Recall, Macro-F1, and Confusion Matrices for classification.
- **Strict Leak-Free Partitioning**: Evaluated on held-out simulation trajectories partitioned strictly by simulation run ID (preventing temporal data leakage).
- **Explainable AI (XAI)**: Feature importance ranking displaying the percentage contribution of plant telemetry features (EC, Cl⁻, basin volume, blowdown, thermal load).

### 4. Control Policy Benchmarking & Water Conservation
- **Three Primary Control Policies**:
  1. *Baseline*: Traditional EC-threshold hysteresis controller.
  2. *Risk-Aware*: Multi-objective controller evaluating conductivity and provisional composite WQRI.
  3. *Water-Use Optimization*: Dynamic conservation throttling operating safely near high-concentration limits.
- **Results**: Achieves **16.7% reduction in total makeup-water consumption** relative to baseline hysteresis while maintaining **100.0% safe operating compliance**.

### 5. Deterministic Synthetic Dataset Generator & Schema Validator
- **Synthetic Data Pipeline**: Generates reproducible multi-scenario datasets across 6 operational regimes (Normal, High Load, Poor Water Quality, Restricted Makeup, Restricted Blowdown, Combined Stress).
- **Clear Labeling**: Every record labeled as `SYNTHETIC` with configurable random seeds.
- **CSV Data Ingestion**: Flexible drag-and-drop CSV uploader with strict schema verification, timestamp monotonicity checks, and physical bounds validation.

### 6. Peer-Reviewed Research Grounding (IEEE / Elsevier)
All modeling principles and evaluation methodologies are grounded in verifiable Elsevier and IEEE literature:
1. **Zhu et al. (2024)**, *Applied Energy* (DOI: [10.1016/j.apenergy.2024.123840](https://doi.org/10.1016/j.apenergy.2024.123840))
2. **Wang et al. (2023)**, *Journal of Process Control* (DOI: [10.1016/j.jprocont.2023.01.004](https://doi.org/10.1016/j.jprocont.2023.01.004))
3. **Chen, Tu, and Yang (2023)**, *Applied Thermal Engineering* (DOI: [10.1016/j.applthermaleng.2023.121047](https://doi.org/10.1016/j.applthermaleng.2023.121047))
4. **Renewable Energy (2019)** (DOI: [10.1016/j.renene.2019.05.127](https://doi.org/10.1016/j.renene.2019.05.127))

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```powershell
# Clone or navigate to the repository
cd c:\Users\Admin\OneDrive\Desktop\hydrotwin_ai

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install requirements (if not already installed)
pip install -r requirements.txt
```

### 2. Launch the Application
```powershell
# Launch Streamlit Research Dashboard
streamlit run app.py
```
*(Or directly: `streamlit run hydrotwin_twin.py`)*

The application will open in your browser at `http://localhost:8501`.

### 3. Run Automated Tests
```powershell
# Run with Pytest (20 automated tests)
pytest

# Or run with Unittest test discovery
python -m unittest discover tests
```

---

## 📁 Repository Structure

```
hydrotwin_ai/
├── app.py                       # Application entry point
├── hydrotwin_twin.py            # Main Streamlit Research Dashboard (9 interactive tabs)
├── requirements.txt             # Project dependencies (Streamlit, Scikit-learn, Plotly, etc.)
├── pytest.ini                   # Pytest configuration
├── RESEARCH_REPORT.md           # Full academic research report with IEEE citations
├── engine/                      # Core computational engine
│   ├── physics.py               # Mass balances, latent heat evaporation, species kinetics
│   ├── models.py                # Water source profiles & default configuration
│   ├── metrics.py               # KPI tracking & control policy comparison
│   ├── scenarios.py             # 6 standardized data-center operational regimes
│   ├── sensitivity.py           # 1D/2D parametric sweep engine
│   ├── dataset_generator.py     # Deterministic synthetic data generator (leak-free splits)
│   ├── data_ingestion.py        # CSV schema validation & boundary checking
│   └── ml_analytics.py          # Scikit-learn predictive ML & explainable AI pipeline
├── visualization/               # Digital twin rendering components
│   ├── twin_3d.py               # Three.js WebGL moving 3D digital twin
│   └── twin_2d.py               # HTML5 Canvas 2D P&ID engineering schematic
├── assets/                      # Offline rendering and export assets
│   ├── three.min.js             # Local Three.js WebGL engine (100% offline)
│   ├── OrbitControls.js         # Camera navigation controls
│   ├── trained_models/          # Serialized scikit-learn models & benchmarks
│   └── data_exports/            # Pre-generated CSV exports & synthetic datasets
├── tests/                       # Automated test suite (20 unit tests)
│   ├── test_simulation.py       # Physics mass balances, zero-flow, bounds clamping
│   ├── test_visualization.py    # HTML generation & scenario simulation tests
│   └── test_ml_and_data.py      # ML holdout metrics, leakage safeguards, schema validation
└── backup/                      # Verifiable pre-modification backups
```

---

## 🔬 Scientific Disclaimers
1. **Illustrative Proxies**: Electrical conductivity (EC) is modeled as an illustrative proportional chloride-proxy. In physical water systems, conductivity depends on multi-ion speciation, temperature, and ionic strength.
2. **Provisional Risk Index**: The Water Quality Risk Index (WQRI) is an engineering composite risk proxy and does not constitute a legally binding regulatory score.
3. **Synthetic Model Fidelity**: Machine learning scores obtained on synthetic simulation data quantify adherence to the mathematical cooling model, not empirical sensor accuracy against a physical facility.
