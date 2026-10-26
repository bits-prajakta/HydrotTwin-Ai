"""
HYDROTWIN-AI: 2D Engineering Process Flow View (P&ID)
Provides an animated, responsive process-and-instrumentation diagram (P&ID)
using native HTML5 Canvas and SVG. Operates completely offline with zero external dependencies.
"""

import json
from typing import Dict, Any
import pandas as pd


def generate_2d_pnd_html(df: pd.DataFrame, config: Dict[str, Any], initial_step: int = 0) -> str:
    """
    Generates a 2D engineering process flow schematic with animated flow pulses,
    dynamic basin water level, animated valve positions, evaporation plumes,
    and synchronized playback controls.
    """
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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HYDROTWIN-AI 2D Process Flow</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Lora', 'Georgia', serif; user-select: none; }}
  body, html {{ width: 100%; height: 100%; overflow: hidden; background: #0f172a; color: #f8fafc; }}
  #container {{ width: 100%; height: 100%; display: flex; flex-direction: column; position: relative; }}
  
  /* Telemetry Top Bar */
  .hud-bar {{
    display: flex; justify-content: space-between; align-items: center;
    background: #1e293b; border-bottom: 1px solid #334155; padding: 8px 18px;
    font-size: 12px; z-index: 10;
  }}
  .hud-cluster {{ display: flex; gap: 10px; align-items: center; }}
  .hud-pill {{
    background: #0f172a; border: 1px solid #38bdf8; border-radius: 6px;
    padding: 4px 10px; display: flex; gap: 6px;
  }}
  .hud-pill .val {{ font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono', monospace; }}
  .hud-pill.alert-normal .val {{ color: #4ade80; }}
  .hud-pill.alert-elevated {{ border-color: #f59e0b; }}
  .hud-pill.alert-elevated .val {{ color: #fbbf24; }}
  .hud-pill.alert-critical {{ border-color: #ef4444; background: #450a0a; }}
  .hud-pill.alert-critical .val {{ color: #f87171; }}

  /* Main Canvas Area */
  #canvas-wrapper {{
    flex: 1; position: relative; width: 100%; height: 100%; overflow: hidden;
  }}
  canvas {{ display: block; width: 100%; height: 100%; }}

  /* Bottom Playback Console */
  .playback-bar {{
    background: #1e293b; border-top: 1px solid #334155; padding: 10px 18px;
    display: flex; flex-direction: column; gap: 8px; z-index: 10;
  }}
  .scrubber-row {{ display: flex; align-items: center; gap: 12px; width: 100%; }}
  .time-slider {{
    flex: 1; -webkit-appearance: none; appearance: none;
    height: 6px; border-radius: 4px; background: #334155; outline: none; cursor: pointer;
  }}
  .time-slider::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none;
    width: 16px; height: 16px; border-radius: 50%;
    background: #38bdf8; border: 2px solid #ffffff; cursor: pointer;
  }}
  .controls-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 12px; }}
  .btn-group {{ display: flex; align-items: center; gap: 8px; }}
  .ctrl-btn {{
    background: #0f172a; border: 1px solid #475569; color: #f8fafc;
    border-radius: 6px; padding: 4px 12px; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
  }}
  .ctrl-btn:hover {{ background: #0284c7; border-color: #38bdf8; }}
  .ctrl-btn.active {{ background: #0284c7; border-color: #38bdf8; }}
  .speed-select {{
    background: #0f172a; border: 1px solid #475569; color: #f8fafc;
    border-radius: 6px; padding: 3px 8px; font-size: 11px; outline: none;
  }}
</style>
</head>
<body>

<div id="container">
  <!-- HUD Header -->
  <div class="hud-bar">
    <div class="hud-cluster">
      <span style="font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;">2D P&ID ENGINEERING VIEW</span>
      <div class="hud-pill"><span>TIME:</span><span class="val" id="hud-time">0.00 h</span></div>
      <div class="hud-pill"><span>BASIN VOL:</span><span class="val" id="hud-vol">0.0 m³</span></div>
      <div class="hud-pill"><span>FILL:</span><span class="val" id="hud-fill">0.0%</span></div>
      <div class="hud-pill"><span>EC PROXY:</span><span class="val" id="hud-ec">0.0 µS/cm</span></div>
      <div class="hud-pill"><span>WQRI:</span><span class="val" id="hud-wqri">0.000</span></div>
    </div>
    <div class="hud-cluster">
      <div class="hud-pill" id="alarm-pill">
        <span>STATUS:</span><span class="val" id="hud-alarm">NORMAL</span>
      </div>
    </div>
  </div>

  <!-- Drawing Canvas -->
  <div id="canvas-wrapper">
    <canvas id="pid-canvas"></canvas>
  </div>

  <!-- Playback Console -->
  <div class="playback-bar">
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
const SIM_DATA = {sim_data_json};
const CONFIG = {config_json};
const totalSteps = SIM_DATA.length - 1;

let currentStep = Math.min(CONFIG.initial_step || 0, totalSteps);
let isPlaying = false;
let playbackSpeed = 1.0;
let lastTimestamp = performance.now();
let stepAccumulator = 0;
let animOffset = 0;

document.getElementById('time-slider').max = totalSteps;
document.getElementById('lbl-max-step').innerText = totalSteps;

const canvas = document.getElementById('pid-canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {{
  const wrapper = document.getElementById('canvas-wrapper');
  canvas.width = wrapper.clientWidth * window.devicePixelRatio;
  canvas.height = wrapper.clientHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawPnd() {{
  const wrapper = document.getElementById('canvas-wrapper');
  const w = wrapper.clientWidth;
  const h = wrapper.clientHeight;
  const data = SIM_DATA[currentStep];
  if (!data) return;

  ctx.clearRect(0, 0, w, h);

  // Background Grid
  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  const gridStep = 40;
  for (let x = 0; x < w; x += gridStep) {{
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }}
  for (let y = 0; y < h; y += gridStep) {{
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }}

  // Coordinate Anchor Ratios
  const srcX = w * 0.08, srcY = h * 0.32, srcW = 120, srcH = 160;
  const basinX = w * 0.38, basinY = h * 0.42, basinW = 220, basinH = 150;
  const dcX = w * 0.78, dcY = h * 0.32, dcW = 150, dcH = 170;
  const ctX = w * 0.38, ctY = h * 0.16, ctW = 220, ctH = 90;
  const sumpX = w * 0.12, sumpY = h * 0.75, sumpW = 100, sumpH = 70;

  // 1. Draw Makeup Source Tank
  ctx.fillStyle = "#164e63";
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 2;
  ctx.strokeRect(srcX, srcY, srcW, srcH);
  ctx.fillRect(srcX, srcY, srcW, srcH);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText("MAKEUP TANK", srcX + 12, srcY + 24);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#94a3b8";
  ctx.fillText(CONFIG.source, srcX + 12, srcY + 44);
  ctx.fillText("Flow: " + data.makeup_flow.toFixed(1) + " m³/h", srcX + 12, srcY + 64);

  // 2. Draw Recirculating Basin
  ctx.fillStyle = "#1e293b";
  ctx.strokeStyle = (data.alarm === 'CRITICAL') ? "#ef4444" : ((data.alarm === 'ELEVATED') ? "#f59e0b" : "#38bdf8");
  ctx.lineWidth = 3;
  ctx.strokeRect(basinX, basinY, basinW, basinH);
  ctx.fillRect(basinX, basinY, basinW, basinH);

  // Dynamic Water Level inside Basin
  const fillFrac = Math.max(0.1, Math.min(1.0, data.fill_pct / 100.0));
  const waterH = fillFrac * (basinH - 8);
  const waterY = basinY + basinH - waterH - 4;

  ctx.fillStyle = (data.alarm === 'CRITICAL') ? "#ef444499" : ((data.alarm === 'ELEVATED') ? "#f59e0b99" : "#0284c7bb");
  ctx.fillRect(basinX + 4, waterY, basinW - 8, waterH);

  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 14px sans-serif";
  ctx.fillText("RECIRCULATING BASIN", basinX + 16, basinY + 26);
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#cbd5e1";
  ctx.fillText("Vol: " + data.volume.toFixed(1) + " / " + CONFIG.capacity.toFixed(0) + " m³", basinX + 16, basinY + 48);
  ctx.fillText("Level: " + data.fill_pct.toFixed(1) + "%", basinX + 16, basinY + 68);
  ctx.fillText("Water Age: " + (data.mean_water_age_h ? data.mean_water_age_h.toFixed(1) : "12.4") + " h", basinX + 16, basinY + 88);

  // 3. Draw Cooling Tower Section
  ctx.fillStyle = "#1e293b";
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 2;
  ctx.strokeRect(ctX, ctY, ctW, ctH);
  ctx.fillRect(ctX, ctY, ctW, ctH);

  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText("COOLING TOWER CELLS", ctX + 24, ctY + 28);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "11px sans-serif";
  ctx.fillText("Evap: " + data.evap_flow.toFixed(2) + " m³/h", ctX + 24, ctY + 50);

  // Evaporation Plume Animation
  if (data.evap_flow > 0) {{
    ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
    for (let p = 0; p < 8; p++) {{
      const px = ctX + 40 + p * 18 + Math.sin(animOffset * 3 + p) * 8;
      const py = ctY - 10 - ((animOffset * 40 + p * 12) % 35);
      ctx.beginPath();
      ctx.arc(px, py, 6 + p % 4, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}

  // 4. Draw Data Center Heat Load Section
  ctx.fillStyle = "#1e293b";
  ctx.strokeStyle = "#f97316";
  ctx.lineWidth = 3;
  ctx.strokeRect(dcX, dcY, dcW, dcH);
  ctx.fillRect(dcX, dcY, dcW, dcH);

  ctx.fillStyle = "#f97316";
  ctx.font = "bold 13px sans-serif";
  ctx.fillText("DATA CENTER", dcX + 16, dcY + 28);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#fed7aa";
  ctx.fillText("IT Load: " + data.thermal_load.toFixed(1) + " MW", dcX + 16, dcY + 52);
  ctx.fillText("Mode: Liquid Cooling", dcX + 16, dcY + 72);

  // 5. Draw Discharge Sump Well
  ctx.fillStyle = "#1e293b";
  ctx.strokeStyle = "#d97706";
  ctx.lineWidth = 2;
  ctx.strokeRect(sumpX, sumpY, sumpW, sumpH);
  ctx.fillRect(sumpX, sumpY, sumpW, sumpH);
  ctx.fillStyle = "#fcd34d";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("DISCHARGE", sumpX + 10, sumpY + 24);
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "11px sans-serif";
  ctx.fillText("Blowdown Bleed", sumpX + 10, sumpY + 44);

  // --- Helper: Draw Animated Piping ---
  function drawPipe(points, color, flowRate, isActive = true) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {{
      ctx.lineTo(points[i].x, points[i].y);
    }}
    ctx.stroke();

    // If flow is positive and active, draw animated moving pulses
    if (flowRate > 0 && isActive) {{
      ctx.fillStyle = "#ffffff";
      const totalLen = getPathLength(points);
      const pulseSpeed = 40 + flowRate * 4;
      const count = Math.max(3, Math.floor(totalLen / 45));

      for (let i = 0; i < count; i++) {{
        const dist = ((animOffset * pulseSpeed) + (i * (totalLen / count))) % totalLen;
        const pt = getPointAlongPath(points, dist);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}
  }}

  function getPathLength(pts) {{
    let len = 0;
    for (let i = 1; i < pts.length; i++) {{
      len += Math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y);
    }}
    return len;
  }}

  function getPointAlongPath(pts, targetDist) {{
    let accumulated = 0;
    for (let i = 1; i < pts.length; i++) {{
      const segLen = Math.hypot(pts[i].x - pts[i-1].x, pts[i].y - pts[i-1].y);
      if (accumulated + segLen >= targetDist) {{
        const r = (targetDist - accumulated) / segLen;
        return {{
          x: pts[i-1].x + (pts[i].x - pts[i-1].x) * r,
          y: pts[i-1].y + (pts[i].y - pts[i-1].y) * r,
        }};
      }}
      accumulated += segLen;
    }}
    return pts[pts.length - 1];
  }}

  // 1. Makeup Line (Source -> Valve -> Basin)
  const makeupPts = [
    {{ x: srcX + srcW, y: srcY + 80 }},
    {{ x: basinX - 60, y: srcY + 80 }},
    {{ x: basinX - 60, y: basinY + 50 }},
    {{ x: basinX, y: basinY + 50 }}
  ];
  drawPipe(makeupPts, "#16a34a", data.makeup_flow, data.makeup_active);

  // 2. Cold Water Supply Line (Basin -> DC)
  const supplyPts = [
    {{ x: basinX + basinW, y: basinY + 110 }},
    {{ x: dcX, y: basinY + 110 }}
  ];
  drawPipe(supplyPts, "#0284c7", 25.0, true);

  // 3. Warm Water Return Line (DC -> Cooling Tower)
  const returnPts = [
    {{ x: dcX + 40, y: dcY }},
    {{ x: dcX + 40, y: ctY + 45 }},
    {{ x: ctX + ctW, y: ctY + 45 }}
  ];
  drawPipe(returnPts, "#ea580c", 25.0, true);

  // 4. Blowdown Line (Basin bottom -> Sump Well)
  const blowdownPts = [
    {{ x: basinX + 40, y: basinY + basinH }},
    {{ x: basinX + 40, y: sumpY + 35 }},
    {{ x: sumpX + sumpW, y: sumpY + 35 }}
  ];
  drawPipe(blowdownPts, "#d97706", data.blowdown_flow, data.blowdown_active);

  // --- Draw Valves ---
  function drawValve(x, y, label, isOpen) {{
    ctx.fillStyle = isOpen ? "#22c55e" : "#64748b";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;

    // Bowtie symbol
    ctx.beginPath();
    ctx.moveTo(x - 12, y - 8);
    ctx.lineTo(x + 12, y + 8);
    ctx.lineTo(x + 12, y - 8);
    ctx.lineTo(x - 12, y + 8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Actuator stem
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y - 12);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y - 14, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "10px sans-serif";
    ctx.fillText(label + ": " + (isOpen ? "OPEN" : "CLOSED"), x - 26, y + 20);
  }}

  drawValve(basinX - 60, srcY + 80, "FCV-101", data.makeup_active);
  drawValve(basinX + 40, sumpY + 35, "BDV-201", data.blowdown_active);

  // --- Draw Sensor Display Station ---
  const sensorX = w * 0.72, sensorY = h * 0.70, sensorW = 220, sensorH = 110;
  ctx.fillStyle = "#1e293b";
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(sensorX, sensorY, sensorW, sensorH);
  ctx.fillRect(sensorX, sensorY, sensorW, sensorH);

  ctx.fillStyle = "#38bdf8";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("ONLINE WATER QUALITY SENSORS", sensorX + 12, sensorY + 22);
  ctx.font = "11px monospace";
  ctx.fillStyle = "#f8fafc";
  ctx.fillText("EC Proxy: " + data.ec.toFixed(0) + " µS/cm", sensorX + 12, sensorY + 42);
  ctx.fillText("WQRI:     " + data.wqri.toFixed(3), sensorX + 12, sensorY + 58);
  ctx.fillText("Chloride: " + data.cl.toFixed(3) + " mg/L", sensorX + 12, sensorY + 74);
  ctx.fillText("Sulfate:  " + data.so4.toFixed(3) + " mg/L", sensorX + 12, sensorY + 90);
}}

function updateHud(step) {{
  const data = SIM_DATA[step];
  if (!data) return;

  document.getElementById('hud-time').innerText = data.time_h.toFixed(2) + ' h';
  document.getElementById('hud-vol').innerText = data.volume.toFixed(1) + ' m³';
  document.getElementById('hud-fill').innerText = data.fill_pct.toFixed(1) + '%';
  document.getElementById('hud-ec').innerText = data.ec.toFixed(0) + ' µS/cm';
  document.getElementById('hud-wqri').innerText = data.wqri.toFixed(3);
  document.getElementById('lbl-step').innerText = step;
  document.getElementById('lbl-event').innerText = data.event;
  document.getElementById('time-slider').value = step;

  const alarmPill = document.getElementById('alarm-pill');
  const hudAlarm = document.getElementById('hud-alarm');
  alarmPill.className = 'hud-pill';
  if (data.alarm === 'CRITICAL') {{
    alarmPill.classList.add('alert-critical');
    hudAlarm.innerText = 'CRITICAL RISK';
  }} else if (data.alarm === 'ELEVATED') {{
    alarmPill.classList.add('alert-elevated');
    hudAlarm.innerText = 'ELEVATED';
  }} else {{
    alarmPill.classList.add('alert-normal');
    hudAlarm.innerText = 'NORMAL';
  }}
}}

function renderLoop(timestamp) {{
  requestAnimationFrame(renderLoop);

  const delta = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;
  animOffset += delta;

  if (isPlaying) {{
    stepAccumulator += delta * playbackSpeed * 2.0;
    if (stepAccumulator >= 1.0) {{
      const stepsToAdvance = Math.floor(stepAccumulator);
      stepAccumulator -= stepsToAdvance;
      currentStep += stepsToAdvance;
      if (currentStep > totalSteps) {{
        currentStep = 0;
      }}
      updateHud(currentStep);
    }}
  }}

  drawPnd();
}}

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
  updateHud(currentStep);
}}

function resetPlayback() {{
  isPlaying = false;
  document.getElementById('btn-play').innerText = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  currentStep = 0;
  stepAccumulator = 0;
  updateHud(currentStep);
}}

function onSliderScrub(val) {{
  currentStep = parseInt(val, 10);
  updateHud(currentStep);
}}

function onSpeedChange(val) {{
  playbackSpeed = parseFloat(val);
}}

updateHud(currentStep);
requestAnimationFrame(renderLoop);
</script>
</body>
</html>
"""
    return html_content
