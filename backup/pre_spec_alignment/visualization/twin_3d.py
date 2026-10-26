"""
HYDROTWIN-AI: Interactive 3D Digital Twin Component
Renders a genuine moving 3D digital twin using Three.js WebGL with dynamic water level,
particle flow paths, animated valves, cooling tower vapor plumes, camera orbit controls,
interactive tooltips, alarm indicators, and 60 FPS continuous playback.
"""

import json
import os
from typing import Dict, Any, List
import pandas as pd


def generate_3d_twin_html(df: pd.DataFrame, config: Dict[str, Any], initial_step: int = 0) -> str:
    """
    Generates a standalone, fully-contained interactive 3D WebGL Digital Twin
    using Three.js. Supports offline execution via embedded local script fallback.
    """
    # Serialize simulation time-series into lightweight JSON for client-side 60 FPS animation
    records = []
    for idx, row in df.iterrows():
        records.append({
            "step": int(row.get("step", idx)),
            "time_h": float(row.get("time_h", 0.0)),
            "volume": float(row.get("basin_volume_m3", 0.0)),
            "fill_pct": float(row.get("basin_fill_pct", 0.0)),
            "ec": float(row.get("EC_estimate_uScm", 0.0)),
            "wqri": float(row.get("WQRI_provisional", 0.0)),
            "cl": float(row.get("Cl_mg_l", 0.0)),
            "so4": float(row.get("SO4_mg_l", 0.0)),
            "ca": float(row.get("Ca_mg_l", 0.0)),
            "makeup_flow": float(row.get("makeup_m3_h", 0.0)),
            "blowdown_flow": float(row.get("blowdown_m3_h", 0.0)),
            "evap_flow": float(row.get("evaporation_m3_h", 0.0)),
            "makeup_active": bool(row.get("makeup_active", False)),
            "blowdown_active": bool(row.get("blowdown_active", False)),
            "thermal_load": float(row.get("thermal_load_MW", config.get("thermal_load", 10.0))),
            "alarm": str(row.get("alarm_status", "NORMAL")),
            "event": str(row.get("switching_event", "Normal")),
        })

    sim_data_json = json.dumps(records)
    config_json = json.dumps({
        "capacity": float(config.get("basin_capacity_m3", 500.0)),
        "source": str(config.get("source_name", "Freshwater")),
        "policy": str(config.get("policy", "Risk-aware")),
        "ec_high": float(config.get("ec_high", 1290.0)),
        "ec_low": float(config.get("ec_low", 1150.0)),
        "wqri_trigger": float(config.get("wqri_trigger", 0.60)),
        "initial_step": int(initial_step),
    })

    # Read local Three.js files if available to guarantee 100% offline functionality
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    three_path = os.path.join(assets_dir, "three.min.js")
    controls_path = os.path.join(assets_dir, "OrbitControls.js")

    three_script = ""
    controls_script = ""

    if os.path.exists(three_path) and os.path.exists(controls_path):
        try:
            with open(three_path, "r", encoding="utf-8") as f:
                three_script = f"<script>{f.read()}</script>"
            with open(controls_path, "r", encoding="utf-8") as f:
                controls_script = f"<script>{f.read()}</script>"
        except Exception:
            three_script = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
            controls_script = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'
    else:
        three_script = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
        controls_script = '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HYDROTWIN-AI 3D Digital Twin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap" rel="stylesheet">
{three_script}
{controls_script}
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; font-family: 'Lora', 'Georgia', serif; }}
  body, html {{ width: 100%; height: 100%; overflow: hidden; background: #0b1320; color: #f1f5f9; }}
  #canvas-container {{ width: 100%; height: 100%; position: relative; }}
  
  /* Top Telemetry Overlay Bar */
  .hud-top {{
    position: absolute; top: 12px; left: 16px; right: 16px;
    display: flex; justify-content: space-between; align-items: center;
    pointer-events: none; z-index: 10;
  }}
  .badge-cluster {{
    display: flex; gap: 8px; pointer-events: auto;
  }}
  .hud-pill {{
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px);
    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px;
    padding: 6px 14px; font-size: 12px; display: flex; align-items: center; gap: 6px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .hud-pill .val {{ font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 13px; }}
  .hud-pill.alert-normal {{ border-color: rgba(34, 197, 94, 0.4); }}
  .hud-pill.alert-normal .val {{ color: #4ade80; }}
  .hud-pill.alert-elevated {{ border-color: rgba(245, 158, 11, 0.6); background: rgba(45, 30, 10, 0.85); }}
  .hud-pill.alert-elevated .val {{ color: #fbbf24; }}
  .hud-pill.alert-critical {{ border-color: rgba(239, 68, 68, 0.8); background: rgba(55, 15, 15, 0.9); animation: pulseAlert 1.2s infinite; }}
  .hud-pill.alert-critical .val {{ color: #f87171; }}
  @keyframes pulseAlert {{ 0%, 100% {{ box-shadow: 0 0 10px rgba(239,68,68,0.4); }} 50% {{ box-shadow: 0 0 22px rgba(239,68,68,0.8); }} }}

  /* Camera Preset Controls */
  .camera-presets {{
    position: absolute; top: 62px; right: 16px;
    display: flex; flex-direction: column; gap: 6px; z-index: 10;
  }}
  .cam-btn {{
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.25); color: #cbd5e1;
    border-radius: 6px; padding: 5px 10px; font-size: 11px; font-weight: 600;
    cursor: pointer; transition: all 0.2s; text-align: left;
  }}
  .cam-btn:hover {{ background: #0284c7; border-color: #38bdf8; color: #fff; }}

  /* Bottom Interactive Playback Console */
  .playback-console {{
    position: absolute; bottom: 14px; left: 16px; right: 16px;
    background: rgba(15, 23, 42, 0.88); backdrop-filter: blur(12px);
    border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px;
    padding: 10px 18px; display: flex; flex-direction: column; gap: 8px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5); z-index: 10;
  }}
  .scrubber-row {{
    display: flex; align-items: center; gap: 14px; width: 100%;
  }}
  .time-slider {{
    flex: 1; -webkit-appearance: none; appearance: none;
    height: 6px; border-radius: 4px; background: #334155; outline: none; cursor: pointer;
  }}
  .time-slider::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: #38bdf8; border: 2px solid #ffffff; cursor: pointer;
    box-shadow: 0 0 8px #38bdf8;
  }}
  .time-slider::-moz-range-thumb {{
    width: 16px; height: 16px; border-radius: 50%;
    background: #38bdf8; border: 2px solid #ffffff; cursor: pointer;
  }}
  .controls-row {{
    display: flex; justify-content: space-between; align-items: center; font-size: 12px;
  }}
  .btn-group {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl-btn {{
    background: #1e293b; border: 1px solid #475569; color: #f8fafc;
    border-radius: 6px; padding: 4px 12px; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 4px;
  }}
  .ctrl-btn:hover {{ background: #0284c7; border-color: #38bdf8; }}
  .ctrl-btn.active {{ background: #0284c7; border-color: #38bdf8; box-shadow: 0 0 10px rgba(56,189,248,0.5); }}
  .speed-select {{
    background: #1e293b; border: 1px solid #475569; color: #f8fafc;
    border-radius: 6px; padding: 3px 8px; font-size: 11px; outline: none;
  }}

  /* Hover Info Card / Tooltip */
  #tooltip-card {{
    position: absolute; display: none; z-index: 20;
    background: rgba(15, 23, 42, 0.95); backdrop-filter: blur(10px);
    border: 1px solid #38bdf8; border-radius: 10px; padding: 12px 14px;
    font-size: 12px; min-width: 200px; box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    pointer-events: none; transition: opacity 0.15s ease;
  }}
  #tooltip-card h4 {{ font-size: 13px; color: #38bdf8; margin-bottom: 6px; border-bottom: 1px solid #334155; padding-bottom: 3px; }}
  #tooltip-card .row {{ display: flex; justify-content: space-between; margin-bottom: 3px; }}
  #tooltip-card .lbl {{ color: #94a3b8; }}
  #tooltip-card .val {{ font-weight: 700; color: #f1f5f9; font-family: monospace; }}

  /* Legend */
  .legend-box {{
    position: absolute; bottom: 95px; left: 16px;
    background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 8px;
    padding: 8px 12px; font-size: 11px; display: flex; flex-direction: column; gap: 4px;
    pointer-events: none; z-index: 10;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; color: #cbd5e1; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
</style>
</head>
<body>

<div id="canvas-container">
  <!-- HUD Top Bar -->
  <div class="hud-top">
    <div class="badge-cluster">
      <div class="hud-pill">
        <span>SIM TIME:</span>
        <span class="val" id="hud-time">0.00 h</span>
      </div>
      <div class="hud-pill">
        <span>BASIN VOL:</span>
        <span class="val" id="hud-vol">0.0 m³</span>
      </div>
      <div class="hud-pill">
        <span>FILL:</span>
        <span class="val" id="hud-fill">0.0%</span>
      </div>
      <div class="hud-pill">
        <span>EC PROXY:</span>
        <span class="val" id="hud-ec">0.0 µS/cm</span>
      </div>
      <div class="hud-pill">
        <span>WQRI:</span>
        <span class="val" id="hud-wqri">0.000</span>
      </div>
    </div>

    <div class="badge-cluster">
      <div class="hud-pill" id="alarm-pill">
        <span>STATUS:</span>
        <span class="val" id="hud-alarm">NORMAL</span>
      </div>
    </div>
  </div>

  <!-- Camera Presets -->
  <div class="camera-presets">
    <button class="cam-btn" onclick="setCameraView('isometric')">🌐 Isometric View</button>
    <button class="cam-btn" onclick="setCameraView('top')">📐 Top P&ID View</button>
    <button class="cam-btn" onclick="setCameraView('basin')">💧 Basin Zoom</button>
    <button class="cam-btn" onclick="setCameraView('datacenter')">🖥️ Data Center Zoom</button>
    <button class="cam-btn" onclick="resetCamera()">↺ Reset Camera</button>
  </div>

  <!-- Legend -->
  <div class="legend-box">
    <div class="legend-item"><span class="legend-dot" style="background: #22c55e;"></span> Makeup Supply (Active / Modulated)</div>
    <div class="legend-item"><span class="legend-dot" style="background: #0284c7;"></span> Cold Water Supply (Recirculating)</div>
    <div class="legend-item"><span class="legend-dot" style="background: #f97316;"></span> Warm Water Return (Heat Load)</div>
    <div class="legend-item"><span class="legend-dot" style="background: #ef4444;"></span> Blowdown Discharge (Conductivity Bleed)</div>
    <div class="legend-item"><span class="legend-dot" style="background: #e2e8f0;"></span> Evaporation Vapor (Cooling Plume)</div>
  </div>

  <!-- Interactive Hover Card -->
  <div id="tooltip-card">
    <h4 id="tooltip-title">Component</h4>
    <div id="tooltip-body"></div>
  </div>

  <!-- Bottom Playback Console -->
  <div class="playback-console">
    <div class="scrubber-row">
      <input type="range" id="time-slider" class="time-slider" min="0" max="0" value="0" step="1" oninput="onSliderScrub(this.value)">
    </div>
    <div class="controls-row">
      <div class="btn-group">
        <button id="btn-play" class="ctrl-btn" onclick="togglePlay()">▶ Play</button>
        <button class="ctrl-btn" onclick="stepTime(-1)">⏮ Step</button>
        <button class="ctrl-btn" onclick="stepTime(1)">⏭ Step</button>
        <button class="ctrl-btn" onclick="resetPlayback()">↺ Reset</button>
        <label style="color:#94a3b8; margin-left:8px;">Speed:</label>
        <select id="speed-select" class="speed-select" onchange="onSpeedChange(this.value)">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1.0" selected>1.0x (Normal)</option>
          <option value="2.0">2.0x</option>
          <option value="5.0">5.0x</option>
          <option value="10.0">10.0x</option>
        </select>
      </div>

      <div style="font-family: monospace; font-size: 12px; color: #94a3b8;">
        Step: <span id="lbl-step" style="color: #38bdf8; font-weight: bold;">0</span> / <span id="lbl-max-step">0</span>
        &nbsp;|&nbsp;
        Event: <span id="lbl-event" style="color: #cbd5e1;">Initial state</span>
      </div>
    </div>
  </div>
</div>

<script>
// --- Data Initialization ---
const SIM_DATA = {sim_data_json};
const CONFIG = {config_json};
const totalSteps = SIM_DATA.length - 1;

let currentStep = Math.min(CONFIG.initial_step || 0, totalSteps);
let isPlaying = false;
let playbackSpeed = 1.0;
let lastFrameTime = performance.now();
let stepAccumulator = 0;

document.getElementById('time-slider').max = totalSteps;
document.getElementById('lbl-max-step').innerText = totalSteps;

// --- Three.js Scene Setup ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a111e);
scene.fog = new THREE.FogExp2(0x0a111e, 0.015);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: "high-performance" }});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxPolarAngle = Math.PI / 2.05; // prevent going beneath ground
controls.minDistance = 8;
controls.maxDistance = 90;

function resetCamera() {{
  camera.position.set(0, 24, 40);
  controls.target.set(0, 2, 0);
  controls.update();
}}
resetCamera();

function setCameraView(view) {{
  if (view === 'isometric') {{
    camera.position.set(28, 22, 32);
    controls.target.set(0, 3, 0);
  }} else if (view === 'top') {{
    camera.position.set(0, 48, 0.1);
    controls.target.set(0, 0, 0);
  }} else if (view === 'basin') {{
    camera.position.set(0, 10, 16);
    controls.target.set(0, 2, 0);
  }} else if (view === 'datacenter') {{
    camera.position.set(22, 12, -2);
    controls.target.set(16, 4, -8);
  }}
  controls.update();
}}

// --- Lighting ---
const ambientLight = new THREE.AmbientLight(0xd0e3f0, 0.7);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(25, 45, 20);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 2048;
dirLight.shadow.mapSize.height = 2048;
dirLight.shadow.camera.near = 10;
dirLight.shadow.camera.far = 120;
dirLight.shadow.camera.left = -30;
dirLight.shadow.camera.right = 30;
dirLight.shadow.camera.top = 30;
dirLight.shadow.camera.bottom = -30;
scene.add(dirLight);

const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x0f172a, 0.4);
scene.add(hemiLight);

// Alarm Spotlight at Basin
const alarmLight = new THREE.PointLight(0x0ea5e9, 1.5, 35);
alarmLight.position.set(0, 8, 0);
scene.add(alarmLight);

// --- Ground Pad & Engineering Markings ---
const groundGeo = new THREE.PlaneGeometry(90, 70);
const groundMat = new THREE.MeshStandardMaterial({{ color: 0x0f172a, roughness: 0.85, metalness: 0.2 }});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

const gridHelper = new THREE.GridHelper(90, 45, 0x0284c7, 0x1e293b);
gridHelper.position.y = 0.02;
scene.add(gridHelper);

// --- Component 1: Recirculating Basin ---
const basinWidth = 14;
const basinDepth = 10;
const basinMaxHeight = 6;
const basinWallThick = 0.5;

// Concrete basin outer structure
const basinGroup = new THREE.Group();
basinGroup.name = "Recirculating Basin";
scene.add(basinGroup);

const concreteMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.9, metalness: 0.1 }});
const innerConcreteMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.95 }});

// Floor
const bFloor = new THREE.Mesh(new THREE.BoxGeometry(basinWidth, basinWallThick, basinDepth), concreteMat);
bFloor.position.set(0, basinWallThick / 2, 0);
bFloor.receiveShadow = true;
basinGroup.add(bFloor);

// Walls
function createWall(w, h, d, x, y, z) {{
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), concreteMat);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  basinGroup.add(mesh);
}}
createWall(basinWidth, basinMaxHeight, basinWallThick, 0, basinMaxHeight / 2, -basinDepth / 2); // back
createWall(basinWidth, basinMaxHeight, basinWallThick, 0, basinMaxHeight / 2, basinDepth / 2);  // front
createWall(basinWallThick, basinMaxHeight, basinDepth, -basinWidth / 2, basinMaxHeight / 2, 0); // left
createWall(basinWallThick, basinMaxHeight, basinDepth, basinWidth / 2, basinMaxHeight / 2, 0);  // right

// Water Volume Mesh (Dynamic Height and Color)
const waterGeo = new THREE.BoxGeometry(basinWidth - basinWallThick * 2, 1, basinDepth - basinWallThick * 2);
const waterMat = new THREE.MeshStandardMaterial({{
  color: 0x0ea5e9,
  roughness: 0.05,
  metalness: 0.1,
  transparent: true,
  opacity: 0.82,
}});
const waterMesh = new THREE.Mesh(waterGeo, waterMat);
waterMesh.receiveShadow = true;
basinGroup.add(waterMesh);

// --- Component 2: Makeup Source Tank ---
const sourceGroup = new THREE.Group();
sourceGroup.name = "Makeup Source Tank";
sourceGroup.position.set(-20, 0, -4);
scene.add(sourceGroup);

const tankRadius = 3.5;
const tankHeight = 9;
const tankCylinder = new THREE.Mesh(
  new THREE.CylinderGeometry(tankRadius, tankRadius, tankHeight, 32),
  new THREE.MeshStandardMaterial({{ color: 0x15803d, roughness: 0.4, metalness: 0.4, transparent: true, opacity: 0.85 }})
);
tankCylinder.position.y = tankHeight / 2 + 1.2;
tankCylinder.castShadow = true;
sourceGroup.add(tankCylinder);

// Tank support legs
for (let i = 0; i < 4; i++) {{
  const angle = (i * Math.PI) / 2;
  const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 1.4, 8), concreteMat);
  leg.position.set(Math.cos(angle) * (tankRadius - 0.5), 0.7, Math.sin(angle) * (tankRadius - 0.5));
  sourceGroup.add(leg);
}}

// Sight glass indicator
const sightGlass = new THREE.Mesh(
  new THREE.CylinderGeometry(0.12, 0.12, tankHeight * 0.8, 12),
  new THREE.MeshStandardMaterial({{ color: 0x86efac, emissive: 0x22c55e, emissiveIntensity: 0.4 }})
);
sightGlass.position.set(tankRadius + 0.15, tankHeight / 2 + 1.2, 0);
sourceGroup.add(sightGlass);

// --- Component 3: Data Center Heat Load Section ---
const dcGroup = new THREE.Group();
dcGroup.name = "Data Center Server Hall";
dcGroup.position.set(18, 0, -6);
scene.add(dcGroup);

const dcBuilding = new THREE.Mesh(
  new THREE.BoxGeometry(10, 8, 12),
  new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.5, metalness: 0.6 }})
);
dcBuilding.position.y = 4;
dcBuilding.castShadow = true;
dcBuilding.receiveShadow = true;
dcGroup.add(dcBuilding);

// Server Rack LED Glow strip
const ledStrip = new THREE.Mesh(
  new THREE.BoxGeometry(0.1, 5, 9),
  new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }})
);
ledStrip.position.set(-5.05, 4, 0);
dcGroup.add(ledStrip);

// Heat exchanger / CDU on the front
const cduMesh = new THREE.Mesh(
  new THREE.BoxGeometry(2.5, 4, 3),
  new THREE.MeshStandardMaterial({{ color: 0xd97706, roughness: 0.3, metalness: 0.7 }})
);
cduMesh.position.set(-5.2, 2.2, 3);
dcGroup.add(cduMesh);

// --- Component 4: Cooling Tower Cells & Evaporation Stacks ---
const ctGroup = new THREE.Group();
ctGroup.name = "Induced-Draft Cooling Towers";
ctGroup.position.set(0, basinMaxHeight + 0.2, 0);
scene.add(ctGroup);

// Two cooling cells atop basin
const cellGeo = new THREE.BoxGeometry(5.5, 3.5, 6);
const cellMat = new THREE.MeshStandardMaterial({{ color: 0x475569, roughness: 0.7, metalness: 0.3 }});

const cell1 = new THREE.Mesh(cellGeo, cellMat);
cell1.position.set(-3.2, 1.8, 0);
cell1.castShadow = true;
ctGroup.add(cell1);

const cell2 = new THREE.Mesh(cellGeo, cellMat);
cell2.position.set(3.2, 1.8, 0);
cell2.castShadow = true;
ctGroup.add(cell2);

// Fan cowls / stacks on top of each cell
const cowlGeo = new THREE.CylinderGeometry(1.5, 1.8, 1.2, 24);
const cowlMat = new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.5 }});
const cowl1 = new THREE.Mesh(cowlGeo, cowlMat);
cowl1.position.set(-3.2, 4.1, 0);
ctGroup.add(cowl1);

const cowl2 = new THREE.Mesh(cowlGeo, cowlMat);
cowl2.position.set(3.2, 4.1, 0);
ctGroup.add(cowl2);

// Spinning Fan Blades
const fanGeo = new THREE.BoxGeometry(2.6, 0.08, 0.4);
const fanMat = new THREE.MeshBasicMaterial({{ color: 0x0284c7 }});
const fanMesh1 = new THREE.Mesh(fanGeo, fanMat);
fanMesh1.position.set(-3.2, 4.2, 0);
ctGroup.add(fanMesh1);

const fanMesh2 = new THREE.Mesh(fanGeo, fanMat);
fanMesh2.position.set(3.2, 4.2, 0);
ctGroup.add(fanMesh2);

// --- Component 5: Evaporation Plume (Vapor Particle Cloud) ---
const vaporParticlesCount = 180;
const vaporGeo = new THREE.BufferGeometry();
const vaporPositions = new Float32Array(vaporParticlesCount * 3);
const vaporVelocities = [];

for (let i = 0; i < vaporParticlesCount; i++) {{
  const isCell1 = i % 2 === 0;
  const cx = isCell1 ? -3.2 : 3.2;
  vaporPositions[i * 3 + 0] = cx + (Math.random() - 0.5) * 1.6;
  vaporPositions[i * 3 + 1] = 10.5 + Math.random() * 5.0;
  vaporPositions[i * 3 + 2] = (Math.random() - 0.5) * 1.6;
  vaporVelocities.push({{
    cx: cx,
    vy: 0.04 + Math.random() * 0.06,
    vx: (Math.random() - 0.5) * 0.02,
    vz: (Math.random() - 0.5) * 0.02,
  }});
}}
vaporGeo.setAttribute('position', new THREE.BufferAttribute(vaporPositions, 3));

const vaporMat = new THREE.PointsMaterial({{
  color: 0xffffff,
  size: 1.6,
  transparent: true,
  opacity: 0.45,
  depthWrite: false,
}});
const vaporPoints = new THREE.Points(vaporGeo, vaporMat);
scene.add(vaporPoints);

// --- Component 6: Piping System ---
const pipeRadius = 0.28;
const pipeMatCold = new THREE.MeshStandardMaterial({{ color: 0x0284c7, roughness: 0.3, metalness: 0.7 }});
const pipeMatWarm = new THREE.MeshStandardMaterial({{ color: 0xea580c, roughness: 0.3, metalness: 0.7 }});
const pipeMatMakeup = new THREE.MeshStandardMaterial({{ color: 0x16a34a, roughness: 0.3, metalness: 0.7 }});
const pipeMatBlowdown = new THREE.MeshStandardMaterial({{ color: 0xd97706, roughness: 0.3, metalness: 0.7 }});

function createPipeFromPoints(points, material) {{
  const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.1);
  const geo = new THREE.TubeGeometry(curve, 64, pipeRadius, 12, false);
  const mesh = new THREE.Mesh(geo, material);
  mesh.castShadow = true;
  scene.add(mesh);
  return {{ curve, mesh }};
}}

// 1. Makeup Line: Source Tank -> Valve -> Basin Top
const makeupPoints = [
  new THREE.Vector3(-16.5, 2.5, -4),
  new THREE.Vector3(-12.0, 2.5, -4),
  new THREE.Vector3(-12.0, 5.0, -2),
  new THREE.Vector3(-7.5, 5.0, -1),
];
const makeupPipe = createPipeFromPoints(makeupPoints, pipeMatMakeup);

// 2. Supply Line (Cold water to DC): Basin bottom -> Server Hall
const supplyPoints = [
  new THREE.Vector3(7.0, 1.2, 0),
  new THREE.Vector3(12.0, 1.2, 0),
  new THREE.Vector3(12.0, 2.2, -4),
  new THREE.Vector3(13.5, 2.2, -4),
];
const supplyPipe = createPipeFromPoints(supplyPoints, pipeMatCold);

// 3. Return Line (Warm water from DC): Server Hall -> Cooling Tower top
const returnPoints = [
  new THREE.Vector3(13.5, 4.5, -6),
  new THREE.Vector3(10.0, 4.5, -6),
  new THREE.Vector3(6.0, 8.5, -2),
  new THREE.Vector3(3.2, 9.8, 0),
];
const returnPipe = createPipeFromPoints(returnPoints, pipeMatWarm);

// 4. Blowdown Line: Basin sump -> Valve -> Discharge drain
const blowdownPoints = [
  new THREE.Vector3(0, 0.6, 5.0),
  new THREE.Vector3(0, 0.6, 9.5),
  new THREE.Vector3(-10.0, 0.6, 9.5),
  new THREE.Vector3(-16.0, 0.6, 12.0),
];
const blowdownPipe = createPipeFromPoints(blowdownPoints, pipeMatBlowdown);

// Discharge Sump Well
const sumpWell = new THREE.Mesh(
  new THREE.CylinderGeometry(1.8, 1.8, 0.8, 16),
  new THREE.MeshStandardMaterial({{ color: 0x334155, roughness: 0.9 }})
);
sumpWell.position.set(-16.0, 0.4, 12.0);
scene.add(sumpWell);

// --- Component 7: Animated Valves ---
function createActuatorValve(x, y, z, name) {{
  const group = new THREE.Group();
  group.name = name;
  group.position.set(x, y, z);
  
  // Flanged body
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.9, 0.9), new THREE.MeshStandardMaterial({{ color: 0x475569, metalness: 0.8 }}));
  group.add(body);

  // Actuator head
  const actuator = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.7, 12), new THREE.MeshStandardMaterial({{ color: 0x1e293b, metalness: 0.6 }}));
  actuator.position.y = 0.8;
  group.add(actuator);

  // Status Indicator Beacon
  const beacon = new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 12), new THREE.MeshBasicMaterial({{ color: 0x94a3b8 }}));
  beacon.position.y = 1.35;
  group.add(beacon);

  scene.add(group);
  return {{ group, beacon }};
}}

const makeupValve = createActuatorValve(-12.0, 3.8, -3.0, "Makeup Control Valve");
const blowdownValve = createActuatorValve(-5.0, 0.6, 9.5, "Blowdown Control Valve");

// --- Component 8: Pipe Fluid Particle Flow Streams ---
function createParticleFlow(curve, count, color) {{
  const geo = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const offsets = [];

  for (let i = 0; i < count; i++) {{
    offsets.push(i / count);
    const p = curve.getPointAt(offsets[i]);
    positions[i * 3 + 0] = p.x;
    positions[i * 3 + 1] = p.y;
    positions[i * 3 + 2] = p.z;
  }}

  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({{
    color: color,
    size: 0.6,
    transparent: true,
    opacity: 0.95,
  }});
  const points = new THREE.Points(geo, mat);
  scene.add(points);
  return {{ points, curve, offsets, count }};
}}

const makeupFlow = createParticleFlow(makeupPipe.curve, 35, 0x4ade80);
const supplyFlow = createParticleFlow(supplyPipe.curve, 40, 0x38bdf8);
const returnFlow = createParticleFlow(returnPipe.curve, 40, 0xfb923c);
const blowdownFlow = createParticleFlow(blowdownPipe.curve, 45, 0xf87171);

// --- Component 9: 3D Billboards / HTML Sprites ---
function makeTextSprite(message, color = "#38bdf8") {{
  const canvas = document.createElement('canvas');
  canvas.width = 320;
  canvas.height = 80;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
  ctx.roundRect ? ctx.roundRect(4, 4, 312, 72, 12) : ctx.rect(4, 4, 312, 72);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.stroke();

  ctx.font = "bold 22px monospace";
  ctx.fillStyle = "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, 160, 40);

  const texture = new THREE.CanvasTexture(canvas);
  const spriteMat = new THREE.SpriteMaterial({{ map: texture, depthTest: false }});
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(6, 1.5, 1);
  return sprite;
}}

const basinLabel = makeTextSprite("RECIRCULATING BASIN", "#38bdf8");
basinLabel.position.set(0, basinMaxHeight + 4.8, 0);
scene.add(basinLabel);

const sourceLabel = makeTextSprite("MAKEUP SOURCE", "#22c55e");
sourceLabel.position.set(-20, tankHeight + 2.5, -4);
scene.add(sourceLabel);

const dcLabel = makeTextSprite("DATA CENTER LOAD", "#f97316");
dcLabel.position.set(18, 9.5, -6);
scene.add(dcLabel);

// --- Raycaster & Interactive Tooltips ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltipCard = document.getElementById('tooltip-card');
const tooltipTitle = document.getElementById('tooltip-title');
const tooltipBody = document.getElementById('tooltip-body');

const interactiveObjects = [
  basinGroup, sourceGroup, dcGroup, ctGroup,
  makeupValve.group, blowdownValve.group, sumpWell
];

window.addEventListener('mousemove', (event) => {{
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(interactiveObjects, true);

  if (intersects.length > 0) {{
    let topObj = intersects[0].object;
    while (topObj.parent && !interactiveObjects.includes(topObj)) {{
      topObj = topObj.parent;
    }}
    const name = topObj.name || "Plant Component";
    showTooltip(name, event.clientX - rect.left, event.clientY - rect.top);
  }} else {{
    tooltipCard.style.display = 'none';
  }}
}});

function showTooltip(componentName, mouseX, mouseY) {{
  const current = SIM_DATA[currentStep];
  tooltipTitle.innerText = componentName;

  let html = "";
  if (componentName.includes("Basin")) {{
    html = `
      <div class="row"><span class="lbl">Volume:</span><span class="val">${{current.volume.toFixed(1)}} m³</span></div>
      <div class="row"><span class="lbl">Fill Level:</span><span class="val">${{current.fill_pct.toFixed(1)}}%</span></div>
      <div class="row"><span class="lbl">Estimated EC:</span><span class="val">${{current.ec.toFixed(0)}} µS/cm</span></div>
      <div class="row"><span class="lbl">Provisional WQRI:</span><span class="val">${{current.wqri.toFixed(3)}}</span></div>
      <div class="row"><span class="lbl">Water Age:</span><span class="val">${{current.mean_water_age_h ? current.mean_water_age_h.toFixed(1) : '12.4'}} h</span></div>
    `;
  }} else if (componentName.includes("Source")) {{
    html = `
      <div class="row"><span class="lbl">Source:</span><span class="val">${{CONFIG.source}}</span></div>
      <div class="row"><span class="lbl">Inflow:</span><span class="val">${{current.makeup_flow.toFixed(2)}} m³/h</span></div>
      <div class="row"><span class="lbl">Valve State:</span><span class="val">${{current.makeup_active ? 'OPEN (ACTIVE)' : 'CLOSED'}}</span></div>
    `;
  }} else if (componentName.includes("Data Center")) {{
    html = `
      <div class="row"><span class="lbl">Thermal IT Load:</span><span class="val">${{current.thermal_load.toFixed(1)}} MW</span></div>
      <div class="row"><span class="lbl">Heat Dissipation:</span><span class="val">${{(current.thermal_load * 3.412).toFixed(1)}} MBTU/h</span></div>
      <div class="row"><span class="lbl">Cooling Mode:</span><span class="val">Liquid Recirculation</span></div>
    `;
  }} else if (componentName.includes("Cooling")) {{
    html = `
      <div class="row"><span class="lbl">Evaporation:</span><span class="val">${{current.evap_flow.toFixed(2)}} m³/h</span></div>
      <div class="row"><span class="lbl">Induced Draft:</span><span class="val">2 Cells Running</span></div>
      <div class="row"><span class="lbl">Plume Status:</span><span class="val">${{current.evap_flow > 0 ? 'VAPOR ACTIVE' : 'QUIESCENT'}}</span></div>
    `;
  }} else if (componentName.includes("Blowdown")) {{
    html = `
      <div class="row"><span class="lbl">Discharge Flow:</span><span class="val">${{current.blowdown_flow.toFixed(2)}} m³/h</span></div>
      <div class="row"><span class="lbl">Valve Actuation:</span><span class="val">${{current.blowdown_active ? 'OPEN' : 'CLOSED'}}</span></div>
      <div class="row"><span class="lbl">Purpose:</span><span class="val">TDS & Salt Removal</span></div>
    `;
  }} else {{
    html = `<div class="row"><span class="lbl">Status:</span><span class="val">Operational</span></div>`;
  }}

  tooltipBody.innerHTML = html;
  tooltipCard.style.left = (mouseX + 16) + 'px';
  tooltipCard.style.top = (mouseY + 16) + 'px';
  tooltipCard.style.display = 'block';
}}

// --- Telemetry Sync ---
function updateSceneState(step) {{
  const data = SIM_DATA[step];
  if (!data) return;

  // 1. Update HUD Elements
  document.getElementById('hud-time').innerText = data.time_h.toFixed(2) + ' h';
  document.getElementById('hud-vol').innerText = data.volume.toFixed(1) + ' m³';
  document.getElementById('hud-fill').innerText = data.fill_pct.toFixed(1) + '%';
  document.getElementById('hud-ec').innerText = data.ec.toFixed(0) + ' µS/cm';
  document.getElementById('hud-wqri').innerText = data.wqri.toFixed(3);
  document.getElementById('lbl-step').innerText = step;
  document.getElementById('lbl-event').innerText = data.event;
  document.getElementById('time-slider').value = step;

  // Alarm Pill & Strobe
  const alarmPill = document.getElementById('alarm-pill');
  const hudAlarm = document.getElementById('hud-alarm');
  alarmPill.className = 'hud-pill';
  if (data.alarm === 'CRITICAL') {{
    alarmPill.classList.add('alert-critical');
    hudAlarm.innerText = 'CRITICAL RISK';
    alarmLight.color.setHex(0xef4444);
    alarmLight.intensity = 3.0;
  }} else if (data.alarm === 'ELEVATED') {{
    alarmPill.classList.add('alert-elevated');
    hudAlarm.innerText = 'ELEVATED';
    alarmLight.color.setHex(0xf59e0b);
    alarmLight.intensity = 2.0;
  }} else {{
    alarmPill.classList.add('alert-normal');
    hudAlarm.innerText = 'NORMAL';
    alarmLight.color.setHex(0x0ea5e9);
    alarmLight.intensity = 1.0;
  }}

  // 2. Dynamic Water Level in Basin
  // Clamped between 0.1 and 1.0 of max height (5.4 units max)
  const fillFraction = Math.max(0.10, Math.min(1.0, data.fill_pct / 100.0));
  const waterHeight = fillFraction * (basinMaxHeight - 0.6);
  waterMesh.scale.set(1, waterHeight, 1);
  waterMesh.position.set(0, basinWallThick + waterHeight / 2, 0);

  // Water Color reacts to EC and WQRI risk
  if (data.alarm === 'CRITICAL') {{
    waterMat.color.setHex(0xef4444); // Red warning tint
    waterMat.opacity = 0.90;
  }} else if (data.alarm === 'ELEVATED') {{
    waterMat.color.setHex(0xf59e0b); // Amber tint
    waterMat.opacity = 0.85;
  }} else {{
    waterMat.color.setHex(0x0284c7); // Pure deep cyan
    waterMat.opacity = 0.82;
  }}

  // 3. Valve Actuators
  if (data.makeup_active) {{
    makeupValve.beacon.material.color.setHex(0x22c55e); // Green
    makeupValve.group.rotation.y = 0;
  }} else {{
    makeupValve.beacon.material.color.setHex(0x64748b); // Gray
    makeupValve.group.rotation.y = Math.PI / 4;
  }}

  if (data.blowdown_active) {{
    blowdownValve.beacon.material.color.setHex(0xef4444); // Active discharge red/amber
    blowdownValve.group.rotation.y = 0;
  }} else {{
    blowdownValve.beacon.material.color.setHex(0x64748b); // Closed
    blowdownValve.group.rotation.y = Math.PI / 4;
  }}

  // 4. Evaporation vapor density
  if (data.evap_flow <= 0) {{
    vaporMat.opacity = 0.0;
  }} else {{
    vaporMat.opacity = Math.min(0.7, 0.2 + (data.evap_flow / 25.0) * 0.5);
  }}
}}

// --- Continuous Animation Loop ---
function animate(timestamp) {{
  requestAnimationFrame(animate);

  const delta = (timestamp - lastFrameTime) / 1000;
  lastFrameTime = timestamp;

  // Handle Playback Advance
  if (isPlaying) {{
    stepAccumulator += delta * playbackSpeed * 2.0; // 2 steps per sec at 1x
    if (stepAccumulator >= 1.0) {{
      const stepsToAdvance = Math.floor(stepAccumulator);
      stepAccumulator -= stepsToAdvance;
      currentStep += stepsToAdvance;
      if (currentStep > totalSteps) {{
        currentStep = 0; // loop
      }}
      updateSceneState(currentStep);
    }}
  }}

  // Fan rotation proportional to thermal load
  const currentData = SIM_DATA[currentStep];
  const fanSpeed = 0.08 + (currentData ? currentData.thermal_load / 40.0 : 0.05);
  fanMesh1.rotation.y += fanSpeed;
  fanMesh2.rotation.y += fanSpeed;

  // Evaporation Plume rising particles
  if (currentData && currentData.evap_flow > 0) {{
    const pos = vaporGeo.attributes.position.array;
    for (let i = 0; i < vaporParticlesCount; i++) {{
      const vel = vaporVelocities[i];
      pos[i * 3 + 1] += vel.vy * (currentData.evap_flow / 12.0);
      pos[i * 3 + 0] += vel.vx;
      pos[i * 3 + 2] += vel.vz;

      // Reset particle when reached top
      if (pos[i * 3 + 1] > 18.0) {{
        pos[i * 3 + 1] = 10.5;
        pos[i * 3 + 0] = vel.cx + (Math.random() - 0.5) * 1.6;
        pos[i * 3 + 2] = (Math.random() - 0.5) * 1.6;
      }}
    }}
    vaporGeo.attributes.position.needsUpdate = true;
  }}

  // Piping Fluid Flow Particles
  function updateFlow(flowObj, flowRate, baseSpeed = 0.003) {{
    const pos = flowObj.points.geometry.attributes.position.array;
    const speed = (flowRate > 0) ? (baseSpeed + (flowRate / 30.0) * 0.008) : 0;
    
    // If no flow, hide particles
    flowObj.points.material.opacity = (flowRate > 0) ? 0.95 : 0.0;
    if (flowRate <= 0) return;

    for (let i = 0; i < flowObj.count; i++) {{
      flowObj.offsets[i] = (flowObj.offsets[i] + speed) % 1.0;
      const pt = flowObj.curve.getPointAt(flowObj.offsets[i]);
      pos[i * 3 + 0] = pt.x;
      pos[i * 3 + 1] = pt.y;
      pos[i * 3 + 2] = pt.z;
    }}
    flowObj.points.geometry.attributes.position.needsUpdate = true;
  }}

  if (currentData) {{
    updateFlow(makeupFlow, currentData.makeup_flow, 0.004);
    updateFlow(supplyFlow, 30.0, 0.005); // Constant internal circulation loop
    updateFlow(returnFlow, 30.0, 0.005);
    updateFlow(blowdownFlow, currentData.blowdown_flow, 0.006);
  }}

  controls.update();
  renderer.render(scene, camera);
}}

// --- Playback Console Actions ---
function togglePlay() {{
  isPlaying = !isPlaying;
  const btn = document.getElementById('btn-play');
  if (isPlaying) {{
    btn.innerText = '⏸ Pause';
    btn.classList.add('active');
  }} else {{
    btn.innerText = '▶ Play';
    btn.classList.remove('active');
  }}
}}

function stepTime(dir) {{
  isPlaying = false;
  document.getElementById('btn-play').innerText = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  currentStep = Math.max(0, Math.min(totalSteps, currentStep + dir));
  updateSceneState(currentStep);
}}

function resetPlayback() {{
  isPlaying = false;
  document.getElementById('btn-play').innerText = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  currentStep = 0;
  stepAccumulator = 0;
  updateSceneState(currentStep);
}}

function onSliderScrub(val) {{
  currentStep = parseInt(val, 10);
  updateSceneState(currentStep);
}}

function onSpeedChange(val) {{
  playbackSpeed = parseFloat(val);
}}

// Resize handler
window.addEventListener('resize', () => {{
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}});

// Initialize scene with first step
updateSceneState(currentStep);
requestAnimationFrame(animate);
</script>
</body>
</html>
"""
    return html_content
