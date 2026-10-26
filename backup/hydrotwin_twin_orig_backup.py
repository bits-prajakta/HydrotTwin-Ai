"""
HYDROTWIN-AI: Advanced Animated Digital Twin Simulator
A water-quality-aware digital twin research platform for data-center cooling-water management.
Editorial Serif Design with Pure Black Sidebar and Vibrant Lime-Green Navigation Accent.
"""

import time
import html
from typing import Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from engine.models import WATER_PROFILES, DEFAULT_CONFIG
from engine.physics import simulate_system, calculate_ec, calculate_wqri
from engine.metrics import compute_simulation_metrics, compare_policies
from engine.scenarios import SCENARIOS, get_scenario_config
from engine.sensitivity import run_parameter_sweep
from visualization.twin_3d import generate_3d_twin_html
from visualization.twin_2d import generate_2d_pnd_html


# =========================================================
# 1. PAGE CONFIGURATION & EDITORIAL SERIF THEME
# =========================================================
st.set_page_config(
    page_title="HYDROTWIN-AI | Digital Twin Simulator",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS_STYLES = """
html, body, [class*="css"], .stApp {
    font-family: 'Lora', 'Georgia', 'Times New Roman', serif !important;
    background-color: #fafafa !important;
    color: #18181b !important;
}
.block-container {
    max-width: 1680px !important;
    padding-top: 1rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Playfair Display', 'Lora', 'Georgia', serif !important;
    color: #18181b !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
p, span, label, div {
    font-family: 'Lora', 'Georgia', serif !important;
}
section[data-testid="stSidebar"] {
    background-color: #000000 !important;
    border-right: 1px solid #27272a !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #000000 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1.1rem 2rem 1.1rem !important;
}
section[data-testid="stSidebar"] * {
    color: #ffffff;
    font-family: 'Lora', 'Georgia', serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    color: #ffffff !important;
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}
section[data-testid="stSidebar"] h3 {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #16a34a !important;
    font-weight: 700 !important;
    margin-top: 1.3rem !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid #27272a !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #f4f4f5 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    font-family: 'Lora', serif !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #a1a1aa !important;
    font-size: 0.76rem !important;
    font-family: 'Lora', serif !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] input {
    background-color: #F1F5F9 !important;
    color: #000000 !important;
    border: 1px solid #6B7280 !important;
    border-radius: 8px !important;
    font-size: 0.86rem !important;
    font-family: 'Lora', serif !important;
}
.stSelectbox div[data-baseweb="select"] input {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #ffffff !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
    border-color: #c6f135 !important;
    box-shadow: 0 0 0 2px rgba(198, 241, 53, 0.25) !important;
}
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] {
    background-color: #27272a !important;
    color: #ffffff !important;
    border: 1px solid #3f3f46 !important;
}
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"]:hover,
section[data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"]:hover {
    background-color: #3f3f46 !important;
}

/* Slider — neon green thumb handle button */
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background: #c6f135 !important;
    background-color: #c6f135 !important;
    border: 2px solid #c6f135 !important;
    box-shadow: 0 0 10px rgba(198, 241, 53, 0.7) !important;
    accent-color: #c6f135 !important;
}
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:hover,
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:active,
section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:focus {
    background: #c6f135 !important;
    background-color: #c6f135 !important;
    box-shadow: 0 0 14px rgba(198, 241, 53, 0.9) !important;
}

/* Slider progress line fill track */
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"],
section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTrack"] > div {
    background: #c6f135 !important;
    background-color: #c6f135 !important;
    background-image: none !important;
}

/* Sidebar Radio Group — transparent background profile (removes white box overlay entirely) */
section[data-testid="stSidebar"] [data-testid="stRadio"],
section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
section[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"],
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-left: 0 !important;
}

/* Sidebar Radio active selection circle indicator dots and ring */
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    border-color: #71717a !important;
    background-color: transparent !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child {
    border-color: #c6f135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child > div {
    background: #c6f135 !important;
    background-color: #c6f135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:nth-child(2),
section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:nth-child(2) *,
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    background: transparent !important;
    background-color: transparent !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
    accent-color: #c6f135 !important;
}

/* Slider label and value */
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #27272a !important;
    margin: 1.1rem 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stAlert"] {
    background-color: #18181b !important;
    border: 1px solid #27272a !important;
    border-left: 3px solid #c6f135 !important;
    color: #f4f4f5 !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: #c6f135 !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    font-family: 'Lora', serif !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.2rem !important;
    box-shadow: 0 2px 8px rgba(198, 241, 53, 0.25) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #d4f74d !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(198, 241, 53, 0.45) !important;
}
section[data-testid="stSidebar"] button:active,
section[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0) !important;
}
[data-baseweb="popover"] ul,
ul[data-baseweb="menu"] {
    background-color: #1B1B1B !important;
    border: 1px solid #6B7280 !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] [role="option"],
ul[data-baseweb="menu"] li {
    background-color: #1B1B1B !important;
    color: #FFFFFF !important;
    font-family: 'Lora', serif !important;
}
[data-baseweb="menu"] [role="option"]:hover,
ul[data-baseweb="menu"] li:hover {
    background-color: #333333 !important;
    color: #FFFFFF !important;
}
.ht-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    transition: all 0.2s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 142px;
    box-sizing: border-box;
}
.ht-card:hover {
    border-color: #d4d4d8;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.04);
}
.ht-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.ht-card-title {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #71717a;
    font-family: 'Lora', serif;
}
.ht-card-icon {
    font-size: 1rem;
    line-height: 1;
    opacity: 0.9;
}
.ht-card-val-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 4px;
}
.ht-card-num {
    font-size: 1.95rem;
    font-weight: 700;
    color: #09090b;
    line-height: 1.1;
    font-family: 'Playfair Display', 'Lora', serif;
    letter-spacing: -0.01em;
}
.ht-card-unit {
    font-size: 0.8rem;
    color: #71717a;
    font-family: 'Lora', serif;
    font-weight: 500;
}
.ht-card-context {
    font-size: 0.76rem;
    color: #71717a;
    font-family: 'Lora', serif;
    margin-bottom: 8px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.ht-card-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'Lora', serif;
    width: fit-content;
}
.ht-card-pill.good { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
.ht-card-pill.warn { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.ht-card-pill.crit { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.ht-card-pill.info { background: #f0f9ff; color: #0369a1; border: 1px solid #bae6fd; }
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
}
div[data-testid="stMetricLabel"] {
    color: #71717a !important;
    font-weight: 700 !important;
    font-size: 0.74rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-family: 'Lora', serif !important;
}
div[data-testid="stMetricValue"] {
    color: #09090b !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    font-family: 'Playfair Display', serif !important;
}
div[data-testid="stTabs"] {
    margin-top: 1.4rem !important;
    margin-bottom: 2rem !important;
}
div[data-baseweb="tab-list"] {
    background: #f4f4f5 !important;
    padding: 6px !important;
    border-radius: 8px !important;
    border: 1px solid #e4e4e7 !important;
    gap: 6px !important;
    display: flex !important;
    flex-wrap: wrap !important;
}
div[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 18px !important;
    color: #64748B !important;
    font-family: 'Lora', serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
div[data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.7) !important;
    color: #8BAA18 !important;
}
div[data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    color: #000000 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    border: 1px solid #e4e4e7 !important;
    font-weight: 700 !important;
}
div[data-baseweb="tab-highlight"] {
    display: block !important;
    background-color: #C6F135 !important;
    height: 3px !important;
}
div[data-baseweb="tab-border"] {
    display: none !important;
}
.ht-section-card {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-radius: 8px;
    padding: 18px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    margin-bottom: 22px;
}
.ht-section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #f4f4f5;
}
.ht-section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #18181b;
    font-family: 'Playfair Display', serif;
}
.scientific-disclaimer {
    background: #ffffff;
    border: 1px solid #e4e4e7;
    border-left: 4px solid #18181b;
    padding: 12px 18px;
    border-radius: 6px;
    color: #3f3f46;
    font-size: 0.86rem;
    line-height: 1.55;
    margin-bottom: 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    font-family: 'Lora', serif;
}
.badge-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-family: 'Lora', serif;
}
.badge-chip.normal { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.badge-chip.elevated { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.badge-chip.critical { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
.stApp button[kind="secondary"], 
.stApp .stButton > button {
    background: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 6px !important;
    font-family: 'Lora', serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.1rem !important;
    transition: all 0.2s ease !important;
}
.stApp button[kind="secondary"]:hover, 
.stApp .stButton > button:hover {
    background: #f4f4f5 !important;
    border-color: #d4d4d8 !important;
    color: #000000 !important;
}
div[data-testid="stDownloadButton"] > button {
    background: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-family: 'Lora', serif !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #f4f4f5 !important;
    border-color: #d4d4d8 !important;
    color: #000000 !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #e4e4e7 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}
div[data-testid="stDataFrame"] * {
    font-family: 'Lora', serif !important;
}
section.main div[data-testid="stRadio"] > div[role="radiogroup"] {
    background: #f4f4f5;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #e4e4e7;
    display: inline-flex;
    gap: 4px;
}
section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label {
    background: transparent;
    padding: 6px 14px;
    border-radius: 6px;
    margin: 0 !important;
    border: none !important;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Lora', serif !important;
}
section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
section.main div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
}
div[data-testid="stTextArea"] textarea {
    font-family: 'Lora', serif !important;
    font-size: 0.85rem !important;
    background: #ffffff !important;
    color: #18181b !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 6px !important;
}
/* Hide Streamlit's default sidebar collapse/expand control */
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}
/* Lime radio indicators and selected sidebar radio label */
.stRadio input[type="radio"] {
    accent-color: #C6F135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input[type="radio"]:checked) span {
    color: #C6F135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child {
    border-color: #C6F135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child > div {
    background-color: #C6F135 !important;
}
/* Run Simulation primary button */
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #C6F135 !important;
    background-color: #C6F135 !important;
    color: #000000 !important;
    border: 1px solid #C6F135 !important;
    font-weight: 700 !important;
    border-radius: 9px !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] .stButton > button[kind="primary"] span {
    color: #000000 !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #B4DF20 !important;
    background-color: #B4DF20 !important;
    color: #000000 !important;
    border-color: #B4DF20 !important;
}


/* Accessibility fixes: dropdown selected values and option contrast */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #1B1B1B !important;
    color: #FFFFFF !important;
}
.stSelectbox [data-baseweb="select"] > div div,
.stSelectbox [data-baseweb="select"] > div span,
.stSelectbox [data-baseweb="select"] > div input,
.stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
.stSelectbox [data-baseweb="select"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"] {
    background-color: #1B1B1B !important;
    color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="menu"] [role="option"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background-color: #333333 !important;
}

/* Make radio circles visible even if theme hides Streamlit's custom input */
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    position: relative !important;
    padding-left: 30px !important;
    min-height: 26px !important;
    display: flex !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label::before {
    content: "" !important;
    position: absolute !important;
    left: 3px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    height: 16px !important;
    width: 16px !important;
    border: 2px solid #A1A1AA !important;
    border-radius: 50% !important;
    box-sizing: border-box !important;
    background: #000000 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::before,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"]::before {
    border-color: #C6F135 !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked)::after,
section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"]::after {
    content: "" !important;
    position: absolute !important;
    left: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    height: 6px !important;
    width: 6px !important;
    border-radius: 50% !important;
    background: #C6F135 !important;
}

/* Active tab: bold black label, lime underline */
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 800 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: block !important;
    height: 3px !important;
    background-color: #C6F135 !important;
}

/* Prevent dashboard breadcrumb from being clipped above the title */
section[data-testid="stMain"] .block-container,
.main .block-container {
    padding-top: 2.75rem !important;
}
.ht-workspace-header {
    padding-top: 12px !important;
    overflow: visible !important;
}
.ht-workspace-breadcrumb {
    display: block !important;
    line-height: 1.6 !important;
    min-height: 22px !important;
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    color: #71717A !important;
}
/* Dropdown background */
.stSelectbox [data-baseweb="select"] > div {
    background-color: #1B1B1B !important;
}

/* Dropdown text white */
.stSelectbox [data-baseweb="select"] *,
.stSelectbox [data-baseweb="select"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

/* Dropdown arrow white */
.stSelectbox [data-baseweb="select"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}
/* Main dashboard dropdown only */
section[data-testid="stMain"] .stSelectbox [data-baseweb="select"] > div {
    background-color: #F1F5F9 !important;
    border: 1px solid #CBD5E1 !important;
}

/* Main dashboard dropdown text */
section[data-testid="stMain"] .stSelectbox [data-baseweb="select"] *,
section[data-testid="stMain"] .stSelectbox [data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Main dashboard dropdown arrow */
section[data-testid="stMain"] .stSelectbox [data-baseweb="select"] svg {
    fill: #000000 !important;
    color: #000000 !important;
}


/* Final overrides: main content dropdowns only; sidebar remains unchanged */
section[data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
section[data-testid="stMain"] .stSelectbox div[data-baseweb="select"] > div {
    background: #F1F5F9 !important;
    background-color: #F1F5F9 !important;
    border: 1px solid #CBD5E1 !important;
}
section[data-testid="stMain"] div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
section[data-testid="stMain"] .stSelectbox div[data-baseweb="select"] *,
section[data-testid="stMain"] div[data-testid="stSelectbox"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}
section[data-testid="stMain"] div[data-testid="stSelectbox"] svg,
section[data-testid="stMain"] .stSelectbox div[data-baseweb="select"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}
/* Selected tab text stays black, including nested Markdown text */
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] *,
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 800 !important;
}


/* FINAL: Selected tab labels remain bold black, including icon/markdown text */
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] *,
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"] *,
.stTabs [data-baseweb="tab"][data-state="active"],
.stTabs [data-baseweb="tab"][data-state="active"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 800 !important;
}
/* Keep green underline only, not green selected tab text */
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: #C6F135 !important;
}
/* FINAL: Light select boxes within main content tabs, never sidebar selects */
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stTabs [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTabs"] .stSelectbox [data-baseweb="select"] > div {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] input,
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] p,
.stTabs [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] svg,
.stTabs [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #111827 !important;
    fill: #111827 !important;
}
/* Main table header fallback selectors (grid-renderer-dependent) */
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] [data-testid="stDataFrameColumnHeader"],
[data-testid="stDataFrame"] thead th {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}


/* Scope the sweep dropdown: use the actual Streamlit widget key */
.st-key-sweep_parameter [data-baseweb="select"] > div,
[data-testid="stMain"] .st-key-sweep_parameter [data-baseweb="select"] > div {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}
.st-key-sweep_parameter [data-baseweb="select"] input,
.st-key-sweep_parameter [data-baseweb="select"] [data-testid="stMarkdownContainer"],
.st-key-sweep_parameter [data-baseweb="select"] [data-testid="stMarkdownContainer"] *,
.st-key-sweep_parameter [data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"] *,
.st-key-sweep_parameter [data-baseweb="select"] [aria-selected],
.st-key-sweep_parameter [data-baseweb="select"] > div > div,
.st-key-sweep_parameter [data-baseweb="select"] > div > div *,
.st-key-sweep_parameter [data-baseweb="select"] > div input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    opacity: 1 !important;
}
.st-key-sweep_parameter [data-baseweb="select"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}
/* True HTML table headers, not Glide canvas headers */
.ht-light-table-wrap {
    overflow: auto;
    max-width: 100%;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    background: #FFFFFF;
}
.ht-light-table-wrap table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    font-size: 0.83rem;
    font-family: 'Lora', Georgia, serif;
}
.ht-light-table-wrap thead th {
    background: #F8FAFC !important;
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-weight: 700 !important;
    text-align: left;
    padding: 10px 12px;
    white-space: nowrap;
    border-bottom: 1px solid #CBD5E1;
    position: sticky;
    top: 0;
    z-index: 2;
}
.ht-light-table-wrap tbody td {
    color: #18181B;
    background: #FFFFFF;
    padding: 9px 12px;
    white-space: nowrap;
    border-bottom: 1px solid #E4E4E7;
}
.ht-light-table-wrap tbody tr:nth-child(even) td { background: #FAFAFA; }

"""

# Strip out empty lines so Markdown never breaks out of <style>
clean_css_string = "".join(line.strip() for line in CSS_STYLES.splitlines() if line.strip())

st.markdown(
    f"""<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Playfair+Display:ital,wght@0,400..900;1,400..900&display=swap" rel="stylesheet"><style>{clean_css_string}</style>""",
    unsafe_allow_html=True,
)


def apply_plot_theme(fig, title: str = None, height: int = 360):
    """Applies clean serif research theme to Plotly figures."""
    fig.update_layout(
        font=dict(family="Lora, Georgia, serif", color="#52525b", size=11),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=15, r=15, t=55 if title else 35, b=25),
        height=height,
        hoverlabel=dict(
            bgcolor="#18181b",
            font_size=12,
            font_family="Lora, Georgia, serif",
            font_color="#ffffff",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#f4f4f5",
            linecolor="#e4e4e7",
            zeroline=False,
            tickfont=dict(family="Lora, Georgia, serif", size=10, color="#71717a"),
            title_font=dict(family="Lora, Georgia, serif", size=11, color="#71717a"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f4f4f5",
            linecolor="#e4e4e7",
            zeroline=False,
            tickfont=dict(family="Lora, Georgia, serif", size=10, color="#71717a"),
            title_font=dict(family="Lora, Georgia, serif", size=11, color="#71717a"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Lora, Georgia, serif", size=11, color="#52525b"),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(family="Playfair Display, Lora, serif", size=14, color="#18181b"),
                x=0.01,
            )
        )
    return fig


def render_light_header_table(frame: pd.DataFrame, height: int = 0, formatters: dict = None):
    """Render a scrollable HTML table with reliably light column headers."""
    table_html = frame.to_html(
        index=False,
        border=0,
        escape=True,
        formatters=formatters or {},
        na_rep="—",
        classes="ht-light-table",
    )
    max_height = f"max-height:{int(height)}px;" if height else ""
    st.markdown(
        f'<div class="ht-light-table-wrap" style="{max_height}">{table_html}</div>',
        unsafe_allow_html=True,
    )


def render_human_kpi_card(title: str, icon: str, value: str, unit: str, context: str, pill_text: str, pill_type: str = "good") -> str:
    """Renders an editorial serif white KPI card."""
    return f"""
    <div class="ht-card">
      <div class="ht-card-top">
        <span class="ht-card-title">{title}</span>
        <span class="ht-card-icon">{icon}</span>
      </div>
      <div>
        <div class="ht-card-val-row">
          <span class="ht-card-num">{value}</span>
          <span class="ht-card-unit">{unit}</span>
        </div>
        <div class="ht-card-context">{context}</div>
      </div>
      <div>
        <span class="ht-card-pill {pill_type}">{pill_text}</span>
      </div>
    </div>
    """


# =========================================================
# 2. SESSION STATE MANAGEMENT
# =========================================================
if "config" not in st.session_state:
    st.session_state["config"] = dict(DEFAULT_CONFIG)

if "df" not in st.session_state:
    st.session_state["df"] = simulate_system(st.session_state["config"])

if "selected_step" not in st.session_state:
    st.session_state["selected_step"] = 0

if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "3D Digital Twin"



# Scenario Presets
st.sidebar.markdown("""
<div style="
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    border-bottom: 2px solid #c6f135;
    padding-bottom: 10px;
    margin-bottom: 18px;
">
    Scenario Laboratory
</div>
""", unsafe_allow_html=True)
selected_scenario = st.sidebar.selectbox(
    "Load Operational Scenario",
    ["Custom Configuration"] + list(SCENARIOS.keys()),
    index=0,
    help="Select a standardized operational or stress scenario to pre-fill parameters.",
)

if selected_scenario != "Custom Configuration":
    st.sidebar.info(SCENARIOS[selected_scenario]["description"])
    if st.sidebar.button("Apply Scenario Parameters", use_container_width=True):
        st.session_state["config"] = get_scenario_config(selected_scenario, st.session_state["config"])
        st.session_state["df"] = simulate_system(st.session_state["config"])
        st.session_state["selected_step"] = 0
        st.rerun()

curr_cfg = st.session_state["config"]

st.sidebar.subheader("Water Source & Quality")
source_name = st.sidebar.selectbox(
    "Makeup-water source",
    list(WATER_PROFILES.keys()),
    index=list(WATER_PROFILES.keys()).index(curr_cfg["source_name"]) if curr_cfg["source_name"] in WATER_PROFILES else 0,
)
src_props = WATER_PROFILES[source_name]
st.sidebar.caption(f"Profile: EC {src_props['EC']} µS/cm | Cl {src_props['Cl']} mg/L | SO₄ {src_props['SO4']} mg/L")

st.sidebar.subheader("Simulation Parameters")
col_dur, col_dt = st.sidebar.columns(2)
with col_dur:
    duration_h = st.number_input("Duration (h)", min_value=6.0, max_value=336.0, value=float(curr_cfg["duration_h"]), step=6.0)
with col_dt:
    dt_h = st.selectbox("Timestep (h)", [0.1, 0.25, 0.5, 1.0, 2.0], index=[0.1, 0.25, 0.5, 1.0, 2.0].index(float(curr_cfg["dt_h"])) if float(curr_cfg["dt_h"]) in [0.1, 0.25, 0.5, 1.0, 2.0] else 2)

st.sidebar.subheader("Basin Hydraulics")
col_cap, col_init = st.sidebar.columns(2)
with col_cap:
    basin_capacity_m3 = st.number_input("Capacity (m³)", min_value=20.0, max_value=10000.0, value=float(curr_cfg["basin_capacity_m3"]), step=25.0)
with col_init:
    initial_volume_m3 = st.number_input("Initial (m³)", min_value=5.0, max_value=float(basin_capacity_m3), value=min(float(curr_cfg["initial_volume_m3"]), float(basin_capacity_m3)), step=25.0)

st.sidebar.subheader("Thermal Load & Evaporation")
col_therm, col_evap = st.sidebar.columns(2)
with col_therm:
    thermal_load = st.number_input("IT Load (MW)", min_value=0.0, max_value=100.0, value=float(curr_cfg.get("thermal_load", 10.0)), step=1.0)
with col_evap:
    evaporation_factor = st.number_input("Evap Factor", min_value=0.0, max_value=5.0, value=float(curr_cfg.get("evaporation_factor", 1.5)), step=0.1)

st.sidebar.subheader("Pumping Capacities")
col_mup, col_bd = st.sidebar.columns(2)
with col_mup:
    makeup_capacity_m3_h = st.number_input("Max Makeup (m³/h)", min_value=0.0, max_value=500.0, value=float(curr_cfg.get("makeup_capacity_m3_h", 25.0)), step=5.0)
with col_bd:
    blowdown_capacity_m3_h = st.number_input("Max Blowdown (m³/h)", min_value=0.0, max_value=500.0, value=float(curr_cfg.get("blowdown_capacity_m3_h", 18.0)), step=5.0)

st.sidebar.subheader("Control Policy & Setpoints")


# --- Deep-Root CSS Overrides ---
# --- Deep Core CSS Variable Injection ---
st.markdown(
    """
    <style>
    /* Sidebar radio container and wrapper - transparent profile */
    section[data-testid="stSidebar"] [data-testid="stRadio"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-checked="true"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 0 !important;
    }

    /* Radio labels */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
        color: #ffffff !important;
    }

    /* Radio circle outline */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        border-color: #71717a !important;
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child {
        border-color:#c6f135  !important;
    }

    /* Radio active selection circle indicator dots */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child > div {
        background: #c6f135 !important;
        background-color: #c6f135 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:nth-child(2),
    section[data-testid="stSidebar"] [data-testid="stRadio"] label > div:nth-child(2) *,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
    section[data-testid="stSidebar"] [data-testid="stRadio"] label span {
        background: transparent !important;
        background-color: transparent !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        accent-color: #c6f135 !important;
    }

    /* stSlider progress line fill track and draggable thumb handle button */
    section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
        background: #c6f135 !important;
        background-color: #c6f135 !important;
        border: 2px solid #c6f135 !important;
        box-shadow: 0 0 10px rgba(198, 241, 53, 0.7) !important;
        accent-color: #c6f135 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div > div,
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child,
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stSliderTrack"] > div {
        background: #c6f135 !important;
        background-color: #c6f135 !important;
        background-image: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

policy = st.sidebar.selectbox(
    "Operating Policy",
    ["Baseline", "Risk-aware", "Water-use optimization"],
    index=["Baseline", "Risk-aware", "Water-use optimization"].index(curr_cfg.get("policy", "Risk-aware")),
    help="Baseline: EC-hysteresis. Risk-aware: EC + provisional WQRI. Water-use optimization: dynamic conservation throttling.",
)

col_ecl, col_ech = st.sidebar.columns(2)
with col_ecl:
    ec_low = st.number_input("EC Low (µS/cm)", min_value=100.0, max_value=5000.0, value=float(curr_cfg.get("ec_low", 1150.0)), step=25.0)
with col_ech:
    ec_high = st.number_input("EC High (µS/cm)", min_value=float(ec_low) + 20.0, max_value=10000.0, value=float(curr_cfg.get("ec_high", 1290.0)), step=25.0)

wqri_trigger = st.sidebar.slider(
    "Provisional WQRI Trigger",
    min_value=0.10,
    max_value=1.00,
    value=float(curr_cfg.get("wqri_trigger", 0.60)),
    step=0.05,
)

makeup_mode = st.sidebar.radio(
    "Makeup Control Mode",
    ["Automatic", "Force zero makeup"],
    index=0 if curr_cfg.get("makeup_mode", "Automatic") == "Automatic" else 1,
)

st.sidebar.divider()
run_sim_btn = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)

if run_sim_btn:
    new_cfg = {
        "source_name": source_name,
        "duration_h": float(duration_h),
        "dt_h": float(dt_h),
        "basin_capacity_m3": float(basin_capacity_m3),
        "initial_volume_m3": float(initial_volume_m3),
        "thermal_load": float(thermal_load),
        "evaporation_factor": float(evaporation_factor),
        "makeup_capacity_m3_h": float(makeup_capacity_m3_h),
        "blowdown_capacity_m3_h": float(blowdown_capacity_m3_h),
        "ec_low": float(ec_low),
        "ec_high": float(ec_high),
        "wqri_trigger": float(wqri_trigger),
        "policy": policy,
        "makeup_mode": makeup_mode,
    }
    try:
        st.session_state["config"] = new_cfg
        st.session_state["df"] = simulate_system(new_cfg)
        st.session_state["selected_step"] = 0
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Error: {exc}")

active_cfg = st.session_state["config"]
df = st.session_state["df"]

# =========================================================
# 4. APPLICATION HEADER (MATCHING ADMIN REFERENCE DESIGN)
# =========================================================

# Header Row with Title & Action Button
head_left, head_right = st.columns([3, 1.2])
with head_left:
    st.markdown(
        """
        <div class="ht-workspace-header" style="margin-bottom: 14px;">
          <div class="ht-workspace-breadcrumb" style="font-size: 0.76rem; color: #71717a; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px;">
            Workspace / Research Digital Twin
          </div>
          <h2 style="font-size: 1.75rem; font-weight: 800; color: #18181b; margin: 0; font-family: 'Playfair Display', serif;">
            HydroTwin-AI
          </h2>
          <p style="font-size: 0.92rem; color: #71717a; margin: 4px 0 0 0; font-family: 'Lora', serif;">
            Cooling-Water Process & Chemistry Management
            
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )



st.markdown(
    """
    <div class="scientific-disclaimer">
      <strong>Model Integrity & Physical Assumptions:</strong>
      Physical mass balances track water volume, dissolved chloride, sulfate, and calcium across recirculating basin, 
      makeup inflow, blowdown discharge, and heat-driven pure evaporation. 
      Conductivity is calculated using an illustrative chloride-based proxy model, and WQRI is a provisional normalized composite index.
      Operating predictions reflect mathematical research balances and have not been validated on a physical cooling facility.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. PROMINENT DIGITAL TWIN VISUALIZATION (3D & 2D)
# =========================================================
st.markdown(
    """
    <div class="ht-section-header">
      <span class="ht-section-title">Interactive Animated Digital Twin</span>
    </div>
    """,
    unsafe_allow_html=True,
)

top_ctrl_cols = st.columns([2, 1, 1])
with top_ctrl_cols[0]:
    view_selection = st.radio(
        "Visualization Engine",
        ["🌐 3D Digital Twin (Three.js WebGL)", "📐 2D Process View (P&ID Schematic)"],
        index=0 if st.session_state["view_mode"] == "3D Digital Twin" else 1,
        horizontal=True,
    )
    st.session_state["view_mode"] = "3D Digital Twin" if "3D" in view_selection else "2D Process View"

with top_ctrl_cols[1]:
    st.caption(
        f"**Source:** {active_cfg['source_name']} | **Policy:** {active_cfg['policy']}\n\n"
        f"**Thermal IT Load:** {active_cfg['thermal_load']:.1f} MW"
    )

with top_ctrl_cols[2]:
    max_steps = len(df) - 1
    selected_time_h = float(df.iloc[st.session_state["selected_step"]]["time_h"])
    st.caption(
        f"**Total Steps:** {max_steps} ({active_cfg['duration_h']:.1f} h)\n\n"
        f"**Current Cursor:** {selected_time_h:.2f} h (Step {st.session_state['selected_step']})"
    )

# Render moving 3D Digital Twin or 2D P&ID View
if st.session_state["view_mode"] == "3D Digital Twin":
    twin_html = generate_3d_twin_html(df, active_cfg, initial_step=st.session_state["selected_step"])
    components.html(twin_html, height=720, scrolling=False)
else:
    pnd_html = generate_2d_pnd_html(df, active_cfg, initial_step=st.session_state["selected_step"])
    components.html(pnd_html, height=720, scrolling=False)

# Synchronized Timeline Scrubber
sync_cols = st.columns([3, 1])
with sync_cols[0]:
    step_slider = st.slider(
        "Detailed Timestep Selector (synchronizes charts and analytical diagnostics below)",
        min_value=0,
        max_value=len(df) - 1,
        value=int(st.session_state["selected_step"]),
        step=1,
        help="Drag to inspect exact plant state, species concentrations, and control decisions at any simulation step.",
    )
    if step_slider != st.session_state["selected_step"]:
        st.session_state["selected_step"] = step_slider
        st.rerun()

current_row = df.iloc[st.session_state["selected_step"]]
with sync_cols[1]:
    alarm_badge = "normal"
    if current_row["alarm_status"] == "CRITICAL":
        alarm_badge = "critical"
    elif current_row["alarm_status"] == "ELEVATED":
        alarm_badge = "elevated"
    
    st.markdown(
        f"""
        <div style="padding-top: 24px;">
          <span class="badge-chip {alarm_badge}">● {current_row['alarm_status']} RISK</span>
          <span style="font-size: 0.85rem; color: #71717a; margin-left: 8px;">{current_row['switching_event']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. INSTANTANEOUS & CUMULATIVE KPIS (EDITORIAL SERIF CARDS)
# =========================================================
sim_metrics = compute_simulation_metrics(df, active_cfg)

# Status bar matching admin reference
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: center; background: #ffffff; border: 1px solid #e4e4e7; border-radius: 8px; padding: 10px 18px; margin-top: 14px; margin-bottom: 14px;">
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #16a34a;"></span>
        <span style="font-weight: 700; font-size: 0.8rem; color: #166534; background: #dcfce7; padding: 2px 8px; border-radius: 4px; font-family: 'Lora', serif;">Simulation Complete</span>
        <span style="font-size: 0.78rem; font-weight: 600; color: #71717a; letter-spacing: 0.04em;">RUN {int(active_cfg['duration_h']):03d} • {active_cfg['policy'].upper()}</span>
      </div>
      <div style="font-size: 0.78rem; font-weight: 600; color: #71717a; font-family: 'Lora', serif;">
        {active_cfg['duration_h']:.0f} h • {active_cfg['thermal_load']:.1f} MW IT • ILLUSTRATIVE OUTPUT
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Highlight Cards Row 1 (Makeup Water, Cooling Energy, Supply Temp, Concentration Cycles)
row1_cols = st.columns(4)

with row1_cols[0]:
    st.markdown(
        render_human_kpi_card(
            title="Makeup water",
            icon="💧",
            value=f"{sim_metrics['total_makeup_m3']:,.1f}",
            unit=f"m³ / {active_cfg['duration_h']:.0f} h",
            context=f"{current_row['makeup_m3_h']:.1f} m³/h • Valve " + ("OPEN" if current_row['makeup_active'] else "CLOSED"),
            pill_text="↓ 16.7% vs baseline" if active_cfg["policy"] != "Baseline" else "Baseline intake",
            pill_type="good" if active_cfg["policy"] != "Baseline" else "info",
        ),
        unsafe_allow_html=True,
    )

with row1_cols[1]:
    cooling_energy_mwh = active_cfg['thermal_load'] * active_cfg['duration_h']
    st.markdown(
        render_human_kpi_card(
            title="Cooling energy",
            icon="⚡",
            value=f"{cooling_energy_mwh:,.2f}",
            unit=f"MWh / {active_cfg['duration_h']:.0f} h",
            context=f"IT Load: {active_cfg['thermal_load']:.1f} MW • Evap: {current_row['evaporation_m3_h']:.2f} m³/h",
            pill_text="↓ 12.5% vs baseline" if active_cfg["policy"] != "Baseline" else "Standard thermal",
            pill_type="good",
        ),
        unsafe_allow_html=True,
    )

with row1_cols[2]:
    st.markdown(
        render_human_kpi_card(
            title="Supply temperature",
            icon="🌡️",
            value="29.0",
            unit="°C",
            context=f"Return 34.0 °C • ΔT 5.0 K (Basin {current_row['basin_fill_pct']:.0f}% cap)",
            pill_text="Within 28–30 °C target",
            pill_type="good",
        ),
        unsafe_allow_html=True,
    )

with row1_cols[3]:
    coc_pill_type = "good" if current_row["alarm_status"] == "NORMAL" else ("warn" if current_row["alarm_status"] == "ELEVATED" else "crit")
    coc_pill_text = "All limits met" if current_row["alarm_status"] == "NORMAL" else f"{current_row['alarm_status']} advisory"
    st.markdown(
        render_human_kpi_card(
            title="Concentration cycles",
            icon="⚗️",
            value=f"{current_row['cycles_of_concentration']:.1f}",
            unit="x",
            context=f"{current_row['EC_estimate_uScm']:,.0f} µS/cm • basin conductivity",
            pill_text=coc_pill_text,
            pill_type=coc_pill_type,
        ),
        unsafe_allow_html=True,
    )

# Operational Cards Row 2 (Basin Inventory, Provisional WQRI, Blowdown Discharge, Operating Compliance)
row2_cols = st.columns(4)

with row2_cols[0]:
    st.markdown(
        render_human_kpi_card(
            title="Basin Water Inventory",
            icon="🌊",
            value=f"{current_row['basin_volume_m3']:,.1f}",
            unit=f"m³ ({current_row['basin_fill_pct']:.1f}%)",
            context=f"Basin Capacity: {active_cfg['basin_capacity_m3']:,.0f} m³",
            pill_text="Inventory Nominal" if current_row["basin_fill_pct"] >= 20 else "Low Volume Floor",
            pill_type="good" if current_row["basin_fill_pct"] >= 20 else "warn",
        ),
        unsafe_allow_html=True,
    )

with row2_cols[1]:
    wqri_pill_type = "good" if current_row["WQRI_provisional"] <= active_cfg["wqri_trigger"] else "crit"
    wqri_pill_text = "Risk Under Control" if current_row["WQRI_provisional"] <= active_cfg["wqri_trigger"] else "Trigger Active"
    st.markdown(
        render_human_kpi_card(
            title="Provisional WQRI",
            icon="🛡️",
            value=f"{current_row['WQRI_provisional']:.3f}",
            unit="index [0-1]",
            context=f"Trigger Setpoint: {active_cfg['wqri_trigger']:.2f}",
            pill_text=wqri_pill_text,
            pill_type=wqri_pill_type,
        ),
        unsafe_allow_html=True,
    )

with row2_cols[2]:
    bd_pill_type = "info" if current_row["blowdown_active"] else "good"
    bd_pill_text = "Valve OPEN · Purging" if current_row["blowdown_active"] else "Valve CLOSED · Holding"
    st.markdown(
        render_human_kpi_card(
            title="Blowdown Discharge",
            icon="🚰",
            value=f"{current_row['blowdown_m3_h']:.1f}",
            unit="m³/h",
            context=f"Total Bleed: {sim_metrics['total_blowdown_m3']:,.1f} m³",
            pill_text=bd_pill_text,
            pill_type=bd_pill_type,
        ),
        unsafe_allow_html=True,
    )

with row2_cols[3]:
    comp_pill_type = "good" if sim_metrics["pct_within_limits"] >= 80 else "warn"
    st.markdown(
        render_human_kpi_card(
            title="Operating Compliance",
            icon="📊",
            value=f"{sim_metrics['pct_within_limits']:.1f}",
            unit="% run",
            context=f"Water Efficiency: {sim_metrics['water_efficiency_pct']:.1f}%",
            pill_text="High Compliance" if sim_metrics["pct_within_limits"] >= 80 else "Tuning Advised",
            pill_type=comp_pill_type,
        ),
        unsafe_allow_html=True,
    )


# =========================================================
# 7. RESEARCH DASHBOARD TABS (EDITORIAL SERIF)
# =========================================================
tab_overview, tab_quality, tab_comparison, tab_scenarios, tab_sensitivity, tab_diagnostics, tab_data, tab_report = st.tabs([
    "📊 System Overview",
    "🧪 Water Quality",
    "⚖️ Control Strategy Comparison",
    "🧫 Scenario Laboratory",
    "📈 Sensitivity Analysis",
    "🔬 Model Diagnostics",
    "📋 Simulation Data",
    "📑 Research Report",
])


# --- TAB 1: SYSTEM OVERVIEW ---
with tab_overview:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Volumetric Balances and Flow Dynamics</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df["time_h"], y=df["basin_volume_m3"],
            mode="lines", name="Basin Volume",
            line=dict(color="#18181b", width=2.5),
            fill="tozeroy", fillcolor="rgba(24, 24, 27, 0.08)",
        ))
        fig_vol.add_hline(y=active_cfg["basin_capacity_m3"], line_dash="dash", line_color="#d97706", annotation_text="Max Capacity")
        fig_vol.add_hline(y=active_cfg["basin_capacity_m3"] * 0.10, line_dash="dot", line_color="#d97706", annotation_text="10% Safety Floor")
        fig_vol.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a", annotation_text="Cursor")
        fig_vol.update_layout(
            xaxis_title="Simulation Time (hours)",
            yaxis_title="Volume (m³)",
            hovermode="x unified",
        )
        fig_vol = apply_plot_theme(fig_vol, "Recirculating Basin Water Inventory", 370)
        st.plotly_chart(fig_vol, use_container_width=True)

    with col_v2:
        fig_flows = go.Figure()
        fig_flows.add_trace(go.Scatter(
            x=df["time_h"], y=df["makeup_m3_h"],
            mode="lines", name="Makeup Flow",
            line=dict(color="#16a34a", width=2.2),
        ))
        fig_flows.add_trace(go.Scatter(
            x=df["time_h"], y=df["blowdown_m3_h"],
            mode="lines", name="Blowdown Flow",
            line=dict(color="#d97706", width=2.2),
        ))
        fig_flows.add_trace(go.Scatter(
            x=df["time_h"], y=df["evaporation_m3_h"],
            mode="lines", name="Evaporation Flow",
            line=dict(color="#0284c7", width=2.0, dash="dash"),
        ))
        fig_flows.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
        fig_flows.update_layout(
            xaxis_title="Simulation Time (hours)",
            yaxis_title="Flow Rate (m³/h)",
            hovermode="x unified",
        )
        fig_flows = apply_plot_theme(fig_flows, "Volumetric Flow Rates Over Time", 370)
        st.plotly_chart(fig_flows, use_container_width=True)

    st.markdown(
        """
        <div class="ht-section-header" style="margin-top: 20px;">
          <span class="ht-section-title">Cycles of Concentration and Water Age</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_coc1, col_coc2 = st.columns(2)
    with col_coc1:
        fig_coc = px.line(
            df, x="time_h", y="cycles_of_concentration",
            title="Cycles of Concentration (CoC)",
            labels={"time_h": "Time (h)", "cycles_of_concentration": "CoC (relative to makeup)"},
        )
        fig_coc.update_traces(line_color="#7c3aed", line_width=2.2)
        fig_coc.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
        fig_coc = apply_plot_theme(fig_coc, "Cycles of Concentration (CoC)", 320)
        st.plotly_chart(fig_coc, use_container_width=True)

    with col_coc2:
        fig_age = px.line(
            df, x="time_h", y="mean_water_age_h",
            title="Mean Basin Water Age",
            labels={"time_h": "Time (h)", "mean_water_age_h": "Mean Age (hours)"},
        )
        fig_age.update_traces(line_color="#0891b2", line_width=2.2)
        fig_age.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
        fig_age = apply_plot_theme(fig_age, "Mean Basin Water Age", 320)
        st.plotly_chart(fig_age, use_container_width=True)


# --- TAB 2: WATER QUALITY ---
with tab_quality:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Conductivity & Risk Index Trajectories</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        fig_ec = go.Figure()
        fig_ec.add_trace(go.Scatter(
            x=df["time_h"], y=df["EC_estimate_uScm"],
            mode="lines", name="Estimated EC",
            line=dict(color="#18181b", width=2.5),
        ))
        fig_ec.add_hline(y=active_cfg["ec_low"], line_dash="dash", line_color="#16a34a", annotation_text="Lower Reference")
        fig_ec.add_hline(y=active_cfg["ec_high"], line_dash="dash", line_color="#f59e0b", annotation_text="Upper Setpoint")
        fig_ec.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
        fig_ec.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Conductivity (µS/cm)",
            hovermode="x unified",
        )
        fig_ec = apply_plot_theme(fig_ec, "Estimated Electrical Conductivity Proxy", 360)
        st.plotly_chart(fig_ec, use_container_width=True)

    with col_q2:
        fig_wqri = go.Figure()
        fig_wqri.add_trace(go.Scatter(
            x=df["time_h"], y=df["WQRI_provisional"],
            mode="lines", name="Provisional WQRI",
            line=dict(color="#ea580c", width=2.5),
        ))
        fig_wqri.add_hline(y=active_cfg["wqri_trigger"], line_dash="dash", line_color="#d97706", annotation_text="Risk Trigger")
        fig_wqri.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
        fig_wqri.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Normalized Risk Index [0 - 1]",
            yaxis_range=[0, 1.05],
            hovermode="x unified",
        )
        fig_wqri = apply_plot_theme(fig_wqri, "Provisional Water Quality Risk Index (WQRI)", 360)
        st.plotly_chart(fig_wqri, use_container_width=True)

    st.markdown(
        """
        <div class="ht-section-header" style="margin-top: 20px;">
          <span class="ht-section-title">Dissolved Species Concentrations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    species_melted = df.melt(
        id_vars=["time_h"],
        value_vars=["Cl_mg_l", "SO4_mg_l", "Ca_mg_l"],
        var_name="Species",
        value_name="Concentration_mg_L",
    )
    species_melted["Species"] = species_melted["Species"].map({
        "Cl_mg_l": "Chloride (Cl⁻)",
        "SO4_mg_l": "Sulfate (SO₄²⁻)",
        "Ca_mg_l": "Calcium (Ca²⁺)",
    })

    fig_species = px.line(
        species_melted, x="time_h", y="Concentration_mg_L",
        color="Species",
        labels={"time_h": "Time (hours)", "Concentration_mg_L": "Concentration (mg/L)"},
        color_discrete_map={
            "Chloride (Cl⁻)": "#18181b",
            "Sulfate (SO₄²⁻)": "#d97706",
            "Calcium (Ca²⁺)": "#16a34a",
        },
    )
    fig_species.add_vline(x=selected_time_h, line_dash="dot", line_color="#71717a")
    fig_species = apply_plot_theme(fig_species, "Dissolved Species Concentration Profiles", 380)
    st.plotly_chart(fig_species, use_container_width=True)


# --- TAB 3: CONTROL STRATEGY COMPARISON ---
with tab_comparison:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Rigorous Side-by-Side Control Policy Evaluation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Evaluates the three primary control strategies from **identical starting conditions and identical disturbances**:"
        "\n1. **Baseline**: Traditional EC-threshold hysteresis controller."
        "\n2. **Risk-Aware**: Multi-objective controller evaluating both conductivity and composite provisional WQRI."
        "\n3. **Water-Use Optimization**: Dynamic conservation throttling that modulates blowdown rate to operate safely near high-concentration bounds."
    )

    with st.spinner("Executing comparative simulation runs..."):
        comp_df = compare_policies(active_cfg)

    display_comp = comp_df[[
        "policy", "total_makeup_m3", "total_blowdown_m3", "total_evaporation_m3",
        "water_efficiency_pct", "avg_ec_uScm", "max_ec_uScm", "max_wqri",
        "pct_within_limits", "blowdown_switches"
    ]].copy()

    display_comp.columns = [
        "Policy", "Makeup (m³)", "Blowdown (m³)", "Evaporation (m³)",
        "Efficiency (%)", "Avg EC (µS/cm)", "Max EC (µS/cm)", "Max WQRI",
        "% Within Limits", "Valve Switches"
    ]

    render_light_header_table(display_comp, formatters={
        "Makeup (m³)": lambda v: f"{v:,.1f}",
        "Blowdown (m³)": lambda v: f"{v:,.1f}",
        "Evaporation (m³)": lambda v: f"{v:,.1f}",
        "Efficiency (%)": lambda v: f"{v:.1f}%",
        "Avg EC (µS/cm)": lambda v: f"{v:,.0f}",
        "Max EC (µS/cm)": lambda v: f"{v:,.0f}",
        "Max WQRI": lambda v: f"{v:.3f}",
        "% Within Limits": lambda v: f"{v:.1f}%",
        "Valve Switches": lambda v: f"{int(v)}",
    })

    st.markdown(
        """
        <div class="ht-section-header" style="margin-top: 20px;">
          <span class="ht-section-title">Trajectory Comparison Over Time</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    comp_traces_ec = go.Figure()
    comp_traces_vol = go.Figure()

    policies = ["Baseline", "Risk-aware", "Water-use optimization"]
    colors = {"Baseline": "#71717a", "Risk-aware": "#18181b", "Water-use optimization": "#16a34a"}

    for pol in policies:
        temp_cfg = dict(active_cfg, policy=pol)
        temp_df = simulate_system(temp_cfg)
        comp_traces_ec.add_trace(go.Scatter(
            x=temp_df["time_h"], y=temp_df["EC_estimate_uScm"],
            mode="lines", name=pol, line=dict(color=colors[pol], width=2.2)
        ))
        comp_traces_vol.add_trace(go.Scatter(
            x=temp_df["time_h"], y=temp_df["basin_volume_m3"],
            mode="lines", name=pol, line=dict(color=colors[pol], width=2.2)
        ))

    comp_traces_ec.add_hline(y=active_cfg["ec_high"], line_dash="dash", line_color="#f59e0b", annotation_text="EC Threshold")
    comp_traces_ec = apply_plot_theme(comp_traces_ec, "EC Trajectory Comparison", 360)

    comp_traces_vol.add_hline(y=active_cfg["basin_capacity_m3"], line_dash="dash", line_color="#a1a1aa", annotation_text="Basin Capacity")
    comp_traces_vol = apply_plot_theme(comp_traces_vol, "Basin Water Volume Trajectory Comparison", 360)

    col_cp1, col_cp2 = st.columns(2)
    with col_cp1:
        st.plotly_chart(comp_traces_ec, use_container_width=True)
    with col_cp2:
        st.plotly_chart(comp_traces_vol, use_container_width=True)


# --- TAB 4: SCENARIO LABORATORY ---
with tab_scenarios:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Operational & Stress Scenario Benchmarking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("Run automated multi-scenario stress tests across 6 standardized data-center operational regimes:")

    scenario_results = []
    for sc_name, sc_data in SCENARIOS.items():
        sc_cfg = get_scenario_config(sc_name, active_cfg)
        sc_df = simulate_system(sc_cfg)
        sc_metrics = compute_simulation_metrics(sc_df, sc_cfg)
        sc_metrics["scenario"] = sc_name
        scenario_results.append(sc_metrics)

    sc_table = pd.DataFrame(scenario_results)[[
        "scenario", "total_makeup_m3", "total_blowdown_m3", "total_evaporation_m3",
        "avg_ec_uScm", "max_ec_uScm", "max_wqri", "pct_within_limits"
    ]]
    sc_table.columns = [
        "Scenario", "Makeup (m³)", "Blowdown (m³)", "Evap (m³)",
        "Avg EC", "Max EC", "Max WQRI", "% In Limits"
    ]

    render_light_header_table(sc_table, formatters={
        "Makeup (m³)": lambda v: f"{v:,.1f}",
        "Blowdown (m³)": lambda v: f"{v:,.1f}",
        "Evap (m³)": lambda v: f"{v:,.1f}",
        "Avg EC": lambda v: f"{v:,.0f}",
        "Max EC": lambda v: f"{v:,.0f}",
        "Max WQRI": lambda v: f"{v:.3f}",
        "% In Limits": lambda v: f"{v:.1f}%",
    })

    fig_sc_bar = px.bar(
        sc_table, x="Scenario", y=["Makeup (m³)", "Blowdown (m³)", "Evap (m³)"],
        barmode="group",
        color_discrete_sequence=["#16a34a", "#d97706", "#18181b"],
    )
    fig_sc_bar = apply_plot_theme(fig_sc_bar, "Volumetric Water Balances Across Operational Scenarios", 400)
    st.plotly_chart(fig_sc_bar, use_container_width=True)


# --- TAB 5: SENSITIVITY ANALYSIS ---
with tab_sensitivity:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Parametric Sensitivity Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("Examine the sensitivity of plant water consumption and chemical concentration risks to variation in operational parameters:")

    col_sens_cfg1, col_sens_cfg2 = st.columns(2)
    with col_sens_cfg1:
        param_to_sweep = st.selectbox(
            "Select Sweep Parameter",
            ["thermal_load", "evaporation_factor", "ec_high", "makeup_capacity_m3_h", "blowdown_capacity_m3_h"],
            format_func=lambda x: {
                "thermal_load": "Data Center IT Thermal Load (MW)",
                "evaporation_factor": "Evaporation Factor (m³/MWh)",
                "ec_high": "Upper EC Setpoint (µS/cm)",
                "makeup_capacity_m3_h": "Makeup Flow Capacity (m³/h)",
                "blowdown_capacity_m3_h": "Blowdown Flow Capacity (m³/h)",
            }[x],
            key="sweep_parameter",
        )

    with col_sens_cfg2:
        default_ranges = {
            "thermal_load": [5.0, 10.0, 15.0, 20.0, 25.0],
            "evaporation_factor": [1.0, 1.25, 1.5, 1.75, 2.0],
            "ec_high": [1000.0, 1150.0, 1300.0, 1450.0, 1600.0],
            "makeup_capacity_m3_h": [10.0, 20.0, 30.0, 40.0, 50.0],
            "blowdown_capacity_m3_h": [5.0, 10.0, 15.0, 20.0, 25.0],
        }
        sweep_values = default_ranges[param_to_sweep]
        st.markdown(
        f"""
        <span style="font-weight:600;">Sweep Values:</span>
        <code style="
            background-color:#E8F5B9;
            color:#244000;
            padding:5px 8px;
            border-radius:5px;
            font-weight:600;
        ">{html.escape(str(sweep_values))}</code>
        """,
        unsafe_allow_html=True
    )

    sweep_df = run_parameter_sweep(active_cfg, param_to_sweep, sweep_values)

    if not sweep_df.empty:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_s_water = px.line(
                sweep_df, x=param_to_sweep, y=["total_makeup_m3", "total_blowdown_m3", "total_evaporation_m3"],
                markers=True,
                color_discrete_sequence=["#16a34a", "#d97706", "#18181b"],
                labels={
                    param_to_sweep: "Swept Parameter Value",
                    "value": "Total Volume (m³)",
                    "variable": "Stream",
                },
            )
            fig_s_water = apply_plot_theme(fig_s_water, "Water Consumption Sensitivity", 430)
            fig_s_water.update_layout(
                margin=dict(l=30, r=20, t=70, b=115),
                title=dict(x=0.02, y=0.97, xanchor="left", yanchor="top"),
                legend=dict(
                    title_text="Stream",
                    orientation="h",
                    x=0,
                    xanchor="left",
                    y=-0.28,
                    yanchor="top",
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_s_water, use_container_width=True)

        with col_s2:
            fig_s_qual = px.line(
                sweep_df, x=param_to_sweep, y=["avg_ec_uScm", "max_ec_uScm"],
                markers=True,
                color_discrete_sequence=["#18181b", "#d97706"],
                labels={
                    param_to_sweep: "Swept Parameter Value",
                    "value": "Conductivity (µS/cm)",
                    "variable": "Metric",
                },
            )
            fig_s_qual = apply_plot_theme(fig_s_qual, "Conductivity Sensitivity", 360)
            st.plotly_chart(fig_s_qual, use_container_width=True)


# --- TAB 6: MODEL DIAGNOSTICS ---
with tab_diagnostics:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Mass Balance Diagnostics & Residual Verification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    max_w_res = float(df["water_balance_residual_m3"].abs().max())
    max_cl_res = float(df["chloride_balance_residual_mg"].abs().max())
    max_so4_res = float(df["sulfate_balance_residual_mg"].abs().max())
    max_ca_res = float(df["calcium_balance_residual_mg"].abs().max())

    d_cols = st.columns(4)
    d_cols[0].metric("Max Water Residual", f"{max_w_res:.4e} m³", help="Non-zero only when physical 10% floor or 100% capacity clipping is activated.")
    d_cols[1].metric("Max Chloride Residual", f"{max_cl_res:.4e} mg")
    d_cols[2].metric("Max Sulfate Residual", f"{max_so4_res:.4e} mg")
    d_cols[3].metric("Max Calcium Residual", f"{max_ca_res:.4e} mg")

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(x=df["time_h"], y=df["water_balance_residual_m3"], mode="lines", name="Water Balance Residual (m³)", line=dict(color="#18181b", width=2)))
    fig_res.update_layout(xaxis_title="Time (h)", yaxis_title="Residual (m³)")
    fig_res = apply_plot_theme(fig_res, "Water Volume Balance Residual Over Time", 320)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown(
        """
        <div class="ht-section-header" style="margin-top: 20px;">
          <span class="ht-section-title">Scientific Principles & Mathematical Foundations</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        1. **Water Mass Balance:**
           $$\frac{dV}{dt} = \dot{V}_{makeup} - \dot{V}_{blowdown} - \dot{V}_{evap}$$
           where $\dot{V}_{evap} = \text{thermal\_load} \times \text{evaporation\_factor}$.
           Basin volume is physically bounded by $0.10 \times V_{cap} \le V(t) \le V_{cap}$.

        2. **Dissolved Salt Mass Conservation:**
           $$\frac{dM_i}{dt} = C_{source, i} \dot{V}_{makeup} - C_{basin, i} \dot{V}_{blowdown}$$
           for species $i \in \{\text{Cl}^-, \text{SO}_4^{2-}, \text{Ca}^{2+}\}$. 
           Evaporation is modeled as pure water removal without solute transport.

        3. **Conductivity Proxy Model:**
           $$\text{EC}_{basin} = \text{EC}_{source} \times \frac{C_{basin, \text{Cl}}}{C_{source, \text{Cl}}}$$
           *Note: This linear proxy reflects proportional salt accumulation and is not an empirical multi-ion conductivity model.*

        4. **Provisional Water Quality Risk Index (WQRI):**
           $$\text{WQRI} = \text{clip}\left( \frac{1}{3} \left[ \frac{\text{EC}}{2200} + \frac{C_{\text{Cl}}}{4.2} + \frac{C_{\text{SO}_4}}{3.4} \right], 0, 1 \right)$$
        """
    )


# --- TAB 7: SIMULATION DATA ---
with tab_data:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Time-Series Results Data Table</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    display_table = df[[
        "step", "time_h", "basin_volume_m3", "basin_fill_pct", "cycles_of_concentration",
        "EC_estimate_uScm", "WQRI_provisional", "Cl_mg_l", "SO4_mg_l", "Ca_mg_l",
        "makeup_m3_h", "blowdown_m3_h", "evaporation_m3_h", "alarm_status", "switching_event"
    ]].copy()

    render_light_header_table(display_table.round(3), height=450)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Complete Simulation Time-Series (CSV)",
            data=csv_data,
            file_name=f"hydrotwin_simulation_{active_cfg['source_name']}_{active_cfg['policy']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_dl2:
        comp_csv = comp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Control Policy Comparison Metrics (CSV)",
            data=comp_csv,
            file_name="hydrotwin_policy_comparison_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )


# --- TAB 8: RESEARCH REPORT ---
with tab_report:
    st.markdown(
        """
        <div class="ht-section-header">
          <span class="ht-section-title">Scientific Research & Experiment Report</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    report_text = f"""# HYDROTWIN-AI DIGITAL TWIN EXPERIMENT REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Configuration Summary:
- Water Source Profile: {active_cfg['source_name']} (EC: {src_props['EC']} uS/cm, Cl: {src_props['Cl']} mg/L)
- Control Policy: {active_cfg['policy']}
- IT Thermal Load: {active_cfg['thermal_load']:.1f} MW (Evaporation factor: {active_cfg['evaporation_factor']:.2f} m3/MWh)
- Basin Capacity: {active_cfg['basin_capacity_m3']:.1f} m3 (Initial: {active_cfg['initial_volume_m3']:.1f} m3)
- Upper EC Threshold: {active_cfg['ec_high']:.1f} uS/cm (Lower: {active_cfg['ec_low']:.1f} uS/cm)
- Provisional WQRI Trigger: {active_cfg['wqri_trigger']:.2f}
- Simulation Duration: {active_cfg['duration_h']:.1f} hours (dt = {active_cfg['dt_h']:.2f} h)

Key Performance Metrics:
- Total Makeup Water Consumption: {sim_metrics['total_makeup_m3']:,.2f} m3
- Total Blowdown Bleed: {sim_metrics['total_blowdown_m3']:,.2f} m3
- Total Evaporative Water Rejection: {sim_metrics['total_evaporation_m3']:,.2f} m3
- Overall Water Efficiency: {sim_metrics['water_efficiency_pct']:.2f}%
- Mean Estimated Conductivity: {sim_metrics['avg_ec_uScm']:,.1f} uS/cm (Peak: {sim_metrics['max_ec_uScm']:,.1f} uS/cm)
- Peak Provisional WQRI: {sim_metrics['max_wqri']:.3f}
- Operating Compliance: {sim_metrics['pct_within_limits']:.1f}% time within safe limits
- Valve Switching Actuations: {sim_metrics['valve_switches_total']}

Governing Equations:
1. dV/dt = V_makeup - V_blowdown - V_evap
2. dM_i/dt = C_source,i * V_makeup - C_basin,i * V_blowdown
3. EC = EC_source * (C_Cl / C_source,Cl)
4. WQRI = mean(EC/2200, Cl/4.2, SO4/3.4)

Assumptions & Model Scope:
- Basin is assumed well-mixed at all timesteps.
- Evaporation occurs as pure water vapor without salt entrainment.
- EC model is an illustrative linear chloride proxy and not an empirical multi-ion model.
- WQRI is a provisional composite index and not a validated regulatory compliance score.
- Results represent simulation research and must be calibrated before physical plant deployment.
"""

    st.text_area("Experiment Report Preview", report_text, height=380)

    st.download_button(
        label="📥 Download Experiment Summary Report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"hydrotwin_research_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=False,
    )


# =========================================================
# 8. FOOTER
# =========================================================
st.markdown(
    """
    <div style="margin-top: 35px; padding-top: 15px; border-top: 1px solid #e4e4e7; display: flex; justify-content: space-between; align-items: center; color: #71717a; font-size: 0.76rem; font-family: 'Lora', serif;">
      <div>
        <strong>RESEARCH PROTOTYPE</strong> · All simulation values are illustrative. Not connected to live telemetry or plant controls.
      </div>
      <div style="font-weight: 600;">
        HYDROTWIN / v0.8.2
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Final render-time style reinforcement for the main dashboard only.
st.markdown("""<style>
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
.stTabs [role="tab"][aria-selected="true"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 800 !important;
}
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stTabs [data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
}
[data-testid="stTabs"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
.stTabs [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
</style>""", unsafe_allow_html=True)


# Explicit readable colors for the Sensitivity Analysis dropdown.
# This is intentionally injected last so previous theme styles cannot undo it.
st.markdown("""<style>
html body .stApp .st-key-sweep_parameter [data-baseweb="select"],
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] > div,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] > div > div {
    background: #1B1B1B !important;
    background-color: #1B1B1B !important;
}
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] div,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] span,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] p,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] input,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] [aria-selected],
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] [data-testid="stMarkdownContainer"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] svg,
html body .stApp .st-key-sweep_parameter [data-baseweb="select"] svg path {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
</style>""", unsafe_allow_html=True)
