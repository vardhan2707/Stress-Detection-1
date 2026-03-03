"""
Stress Detection Dashboard — Streamlit App
Interactive prediction UI + results viewer for the Personalized ML project.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import joblib
from datetime import datetime, timedelta
import random
import hashlib
import io
import urllib.parse
import base64
import sqlite3

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "Final_CSVs")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
MODEL_DIR = os.path.join(BASE_DIR, "models")
STRESS_LOG_PATH = os.path.join(CSV_DIR, "stress_log.csv")
STRESS_DB_PATH = os.path.join(BASE_DIR, "stress_history.db")

# Ensure CSV directory exists for local persistence
os.makedirs(CSV_DIR, exist_ok=True)


def _get_secret_or_env(key, default=""):
    """Get value from st.secrets or os.environ. Avoids StreamlitSecretNotFoundError when no secrets file exists."""
    try:
        val = st.secrets.get(key, os.environ.get(key, default))
        return val if val is not None else os.environ.get(key, default)
    except Exception:
        return os.environ.get(key, default)


def estimate_scl_from_hr_rmssd(hr, rmssd):
    """
    Estimate SCL (skin conductance) when smartwatch does not provide it.
    Higher HR + lower RMSSD → higher SCL (more stress).
    Returns value in µS, typical range 1–25.
    """
    # Base SCL at rest
    base = 5.0
    # HR contribution: elevated HR increases SCL
    hr_factor = max(-5, min(10, (hr - 70) / 80 * 10))
    # RMSSD contribution: lower RMSSD (less variability) increases SCL
    rmssd_factor = max(-5, min(12, (40 - rmssd) / 60 * 10))
    scl = base + hr_factor + rmssd_factor
    return float(np.clip(scl, 1.0, 25.0))


def estimate_rmssd_from_hr(hr):
    """
    Estimate RMSSD (ms) from heart rate when HRV is unavailable.
    Higher HR generally corresponds to lower RMSSD.
    """
    hr_value = float(np.clip(hr, 40.0, 200.0))
    rmssd = 95.0 - 0.7 * hr_value
    return float(np.clip(rmssd, 10.0, 120.0))

st.set_page_config(
    page_title="Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Force sidebar to be visible and expanded (applied early, before authentication check)
st.markdown("""
<style>
    /* Ensure sidebar is always visible when authenticated - applied globally */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
        transform: translateX(0) !important;
        width: 21rem !important;
        min-width: 21rem !important;
    }
    /* Show sidebar toggle button */
    [data-testid="collapsedControl"],
    button[data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Hide Streamlit deploy button and toolbar */
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }

    /* Ensure sidebar is visible when authenticated - IMPORTANT: Must override login page CSS */
    [data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 21rem !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        display: block !important;
    }
    /* Ensure sidebar content is visible */
    [data-testid="stSidebar"] > div {
        display: block !important;
        visibility: visible !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7b68ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -1px;
    }

    .sub-title {
        text-align: center;
        color: #8892a4;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid rgba(0, 212, 170, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 212, 170, 0.1);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #00d4aa;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    .prediction-box {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .stress-high {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 120, 50, 0.1) 100%);
        border: 2px solid rgba(255, 75, 75, 0.4);
    }

    .stress-moderate {
        background: linear-gradient(135deg, rgba(255, 200, 50, 0.15) 0%, rgba(255, 180, 30, 0.1) 100%);
        border: 2px solid rgba(255, 200, 50, 0.4);
    }

    .stress-low {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(0, 180, 216, 0.1) 100%);
        border: 2px solid rgba(0, 212, 170, 0.4);
    }

    .pred-label {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .pred-confidence {
        font-size: 1.1rem;
        color: #8892a4;
    }

    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fafafa;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 170, 0.2);
    }

    .sidebar .sidebar-content {
        background: #0e1117;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
    }

    .dataset-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin: 2px;
    }

    .badge-lab {
        background: rgba(123, 104, 238, 0.2);
        color: #7b68ee;
        border: 1px solid rgba(123, 104, 238, 0.3);
    }

    .badge-wild {
        background: rgba(0, 180, 216, 0.2);
        color: #00b4d8;
        border: 1px solid rgba(0, 180, 216, 0.3);
    }

    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .feature-input-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 1px solid rgba(0, 212, 170, 0.1);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .about-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #1e2636 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .analytics-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #fafafa;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(123, 104, 238, 0.3);
    }

    .alert-banner {
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .alert-warning {
        background: linear-gradient(135deg, rgba(255, 200, 50, 0.12) 0%, rgba(255, 120, 50, 0.08) 100%);
        border: 1px solid rgba(255, 200, 50, 0.35);
    }

    .alert-danger {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.12) 0%, rgba(255, 120, 50, 0.08) 100%);
        border: 1px solid rgba(255, 75, 75, 0.35);
    }

    .alert-success {
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.12) 0%, rgba(0, 180, 216, 0.08) 100%);
        border: 1px solid rgba(0, 212, 170, 0.35);
    }

    .recovery-score-box {
        background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
        border: 2px solid rgba(123, 104, 238, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(123, 104, 238, 0.1);
    }

    .recovery-score-value {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7b68ee 0%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .trend-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .trend-up {
        background: rgba(255, 75, 75, 0.15);
        color: #ff4b4b;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }

    .trend-down {
        background: rgba(0, 212, 170, 0.15);
        color: #00d4aa;
        border: 1px solid rgba(0, 212, 170, 0.3);
    }

    .trend-stable {
        background: rgba(255, 200, 50, 0.15);
        color: #ffc832;
        border: 1px solid rgba(255, 200, 50, 0.3);
    }

    /* ── Predict Stress page: Plus Jakarta / Syne / sci-style cards ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    .pg-label { font-family:'JetBrains Mono',monospace; font-size:11px; color:#0e7490; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; }
    .pg-title { font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#f0f6ff; margin-bottom:6px; }
    .pg-sub { font-size:14px; color:#8ba3c7; margin-bottom:28px; }
    .stat-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:28px; }
    .stat-card { background:#141c2e; border:1px solid #1e2d45; border-radius:14px; padding:22px 24px; position:relative; overflow:hidden; transition:all .25s; }
    .stat-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--ca),transparent); }
    .stat-card::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at top center,var(--cg),transparent 70%); }
    .stat-card:hover { border-color:var(--ca); transform:translateY(-2px); box-shadow:0 8px 32px rgba(0,0,0,.3); }
    .stat-card.c1{--ca:#22d3ee;--cg:rgba(34,211,238,.06)} .stat-card.c2{--ca:#4ade80;--cg:rgba(74,222,128,.06)} .stat-card.c3{--ca:#fbbf24;--cg:rgba(251,191,36,.06)}
    .stat-lbl { font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d5478; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px; }
    .stat-val { font-family:'Syne',sans-serif; font-size:26px; font-weight:700; color:var(--ca); margin-bottom:4px; }
    .stat-meta { font-size:12px; color:#3d5478; font-family:'JetBrains Mono',monospace; }
    .sec-card { background:#141c2e; border:1px solid #1e2d45; border-radius:14px; overflow:hidden; margin-bottom:14px; }
    .sec-head { padding:16px 22px; border-bottom:1px solid #1e2d45; display:flex; align-items:center; gap:12px; background:rgba(34,211,238,.02); }
    .sec-icon { width:32px; height:32px; border-radius:8px; background:rgba(34,211,238,.08); border:1px solid rgba(34,211,238,.2); display:flex; align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }
    .sec-title { font-family:'Syne',sans-serif; font-size:14px; font-weight:700; color:#f0f6ff; }
    .sec-sub { font-size:12px; color:#3d5478; margin-top:2px; }
    .result-box { background:#141c2e; border:1px solid #1e2d45; border-radius:14px; padding:28px 24px; text-align:center; position:relative; overflow:hidden; min-height:220px; display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:14px; }
    .result-box.active { border-color:var(--rc); }
    .result-box.active::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--rc),transparent); }
    .result-box.active::after { content:''; position:absolute; inset:0; background:radial-gradient(ellipse at top center,var(--rg),transparent 65%); }
    .res-emoji { font-size:42px; margin-bottom:8px; position:relative; z-index:1; }
    .res-score { font-family:'Syne',sans-serif; font-size:58px; font-weight:800; line-height:1; position:relative; z-index:1; }
    .res-level { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; letter-spacing:1px; text-transform:uppercase; margin-top:4px; position:relative; z-index:1; }
    .res-bar-wrap { width:100%; margin-top:16px; position:relative; z-index:1; }
    .res-bar-track { height:6px; background:#1e2d45; border-radius:3px; overflow:hidden; }
    .res-bar-fill { height:100%; border-radius:3px; }
    .res-empty-icon { font-size:36px; opacity:.3; margin-bottom:10px; }
    .res-empty-text { font-size:13px; color:#3d5478; line-height:1.7; }
    .bd-card { background:#141c2e; border:1px solid #1e2d45; border-radius:14px; padding:18px; margin-bottom:14px; }
    .bd-title { font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d5478; letter-spacing:2px; text-transform:uppercase; margin-bottom:14px; }
    .bd-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid #1e2d45; font-size:13px; }
    .bd-row:last-child { border-bottom:none; }
    .bd-name { color:#8ba3c7; }
    .bd-val { font-family:'JetBrains Mono',monospace; font-size:12px; color:#22d3ee; }
    .ref-card { background:#141c2e; border:1px solid #1e2d45; border-radius:14px; overflow:hidden; }
    .ref-head { padding:14px 18px; border-bottom:1px solid #1e2d45; font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#8ba3c7; }
    .ref-body { padding:14px 18px; display:flex; flex-direction:column; gap:10px; }
    .ref-row { display:flex; justify-content:space-between; align-items:center; font-size:12px; }
    .ref-sig { color:#8ba3c7; font-weight:500; }
    .ref-rng { font-family:'JetBrains Mono',monospace; font-size:11px; padding:3px 9px; border-radius:20px; }
    .ref-rng.cy { background:rgba(34,211,238,.06); color:#22d3ee; border:1px solid rgba(34,211,238,.15); }
    .ref-rng.am { background:rgba(251,191,36,.06); color:#fbbf24; border:1px solid rgba(251,191,36,.15); }
    .ref-rng.gr { background:rgba(74,222,128,.06); color:#4ade80; border:1px solid rgba(74,222,128,.15); }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
<style>
  /* ─────────────────────────────────────────────────────────────────────────
     Futuristic AI Dashboard Theme (UI only)
     Minimal Apple-like + neon cyber styling via CSS overrides.
     ───────────────────────────────────────────────────────────────────────── */

  :root{
    --bg0:#050812;
    --bg1:#070b17;
    --bg2:#0b1430;
    --glass: rgba(255,255,255,0.06);
    --glass-2: rgba(255,255,255,0.085);
    --border: rgba(0,229,255,0.18);
    --border-2: rgba(124,58,237,0.18);
    --accent:#00e5ff;       /* neon cyan */
    --accent-2:#7c3aed;     /* neon violet */
    --text:#e8eef9;
    --muted:#9aa7b6;
    --shadow: 0 18px 60px rgba(0,0,0,0.55);
  }

  html, body, [class*="css"]{
    color: var(--text);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* App background + subtle grid overlay */
  div[data-testid="stAppViewContainer"]{
    background:
      radial-gradient(1200px 700px at 20% 10%, rgba(0,229,255,0.13) 0%, rgba(0,229,255,0.0) 55%),
      radial-gradient(900px 600px at 85% 20%, rgba(124,58,237,0.16) 0%, rgba(124,58,237,0.0) 55%),
      linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 55%, var(--bg2) 130%);
  }

  div[data-testid="stAppViewContainer"]::before{
    content:"";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
      linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
    background-size: 44px 44px;
    opacity: 0.16;
    mix-blend-mode: overlay;
  }

  /* Content width / spacing */
  .block-container{
    padding-top: 1.35rem;
    padding-bottom: 2.5rem;
  }

  /* Sidebar: modern glass panel */
  section[data-testid="stSidebar"]{
    background:
      radial-gradient(900px 600px at 20% 0%, rgba(0,229,255,0.10) 0%, rgba(0,229,255,0) 60%),
      linear-gradient(180deg, rgba(12,16,32,0.92) 0%, rgba(8,12,24,0.88) 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 12px 0 50px rgba(0,0,0,0.45);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
  }
  section[data-testid="stSidebar"]::after{
    content:"";
    position:absolute;
    inset:0;
    pointer-events:none;
    background:
      linear-gradient(transparent, rgba(0,229,255,0.05) 40%, transparent),
      linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: auto, 52px 52px;
    opacity: .18;
  }

  /* Links */
  a{ color: var(--accent) !important; }
  a:hover{ filter: drop-shadow(0 0 10px rgba(0,229,255,0.35)); }

  /* Titles */
  .main-title{
    font-size: 3.05rem !important;
    letter-spacing: -0.9px !important;
    background: linear-gradient(135deg, var(--accent) 0%, #7af4ff 35%, var(--accent-2) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-shadow: 0 0 22px rgba(0,229,255,0.18);
  }
  .sub-title{
    color: var(--muted) !important;
    max-width: 70ch;
    margin-left: auto;
    margin-right: auto;
  }

  .section-header{
    font-weight: 700 !important;
    letter-spacing: -0.3px;
    color: var(--text) !important;
    position: relative;
  }
  .section-header::after{
    content:"";
    display:block;
    height: 2px;
    margin-top: 10px;
    width: 100%;
    background: linear-gradient(90deg, rgba(0,229,255,0.75), rgba(124,58,237,0.35), transparent);
    border-radius: 999px;
    opacity: .65;
  }

  /* Glassmorphism cards (reuse existing classes) */
  .metric-card,
  .prediction-box,
  .feature-input-card,
  .about-card,
  .device-card,
  .rt-card,
  .rt-chart-card{
    background: linear-gradient(135deg, rgba(255,255,255,0.07), rgba(255,255,255,0.035)) !important;
    border: 1px solid rgba(0,229,255,0.18) !important;
    box-shadow: var(--shadow) !important;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
  }

  .metric-card:hover,
  .device-card:hover,
  .rt-card:hover{
    box-shadow: 0 26px 80px rgba(0,0,0,0.62), 0 0 0 1px rgba(0,229,255,0.12) !important;
    filter: drop-shadow(0 0 22px rgba(0,229,255,0.12));
    transform: translateY(-2px);
  }

  .metric-value{
    color: var(--accent) !important;
    text-shadow: 0 0 18px rgba(0,229,255,0.12);
  }
  .metric-label{
    color: var(--muted) !important;
  }

  /* Prediction variants: keep existing color semantics but add glow */
  .stress-low{ border-color: rgba(0,229,255,0.22) !important; }
  .stress-moderate{ border-color: rgba(255,200,50,0.30) !important; }
  .stress-high{ border-color: rgba(255,75,75,0.32) !important; }

  /* Buttons */
  .stButton>button,
  button[kind="primary"]{
    border-radius: 14px !important;
    border: 1px solid rgba(0,229,255,0.28) !important;
    background: linear-gradient(135deg, rgba(0,229,255,0.95), rgba(124,58,237,0.88)) !important;
    color: #07101a !important;
    font-weight: 700 !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    box-shadow: 0 12px 34px rgba(0,229,255,0.12);
  }
  .stButton>button:hover{
    transform: translateY(-1px);
    box-shadow: 0 18px 60px rgba(0,229,255,0.18);
    filter: drop-shadow(0 0 14px rgba(0,229,255,0.22));
  }
  .stButton>button:active{
    transform: translateY(0px) scale(0.99);
  }

  /* Inputs */
  .stTextInput input,
  .stNumberInput input,
  .stSelectbox div[data-baseweb="select"] > div,
  .stTextArea textarea{
    background: rgba(5,8,18,0.65) !important;
    border: 1px solid rgba(148,163,184,0.20) !important;
    color: var(--text) !important;
    border-radius: 14px !important;
  }
  .stSelectbox svg{ color: var(--muted) !important; }

  /* Tabs */
  button[role="tab"]{
    border-radius: 999px !important;
    transition: background .2s ease, box-shadow .2s ease, border-color .2s ease;
  }
  button[role="tab"][aria-selected="true"]{
    background: rgba(0,229,255,0.10) !important;
    border: 1px solid rgba(0,229,255,0.30) !important;
    box-shadow: 0 10px 30px rgba(0,229,255,0.10);
  }

  /* Sidebar nav radio tweaks */
  section[data-testid="stSidebar"] div[role="radiogroup"] label{
    border-radius: 12px !important;
    padding: 8px 10px !important;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
    background: rgba(0,229,255,0.08) !important;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.14);
  }

  /* Top status bar */
  .ai-topbar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 1rem;
    padding: 0.55rem 0.9rem;
    margin: 0.1rem 0 1.2rem 0;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border: 1px solid rgba(0,229,255,0.16);
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
  }
  .ai-crumb{
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(0,229,255,0.75);
    display:flex;
    align-items:center;
    gap: .45rem;
    white-space: nowrap;
  }
  .ai-status{
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: rgba(232,238,249,0.78);
    display:flex;
    align-items:center;
    gap: .5rem;
    white-space: nowrap;
  }
  .ai-dot{
    width:8px;height:8px;border-radius:999px;
    background: var(--accent);
    box-shadow: 0 0 0 0 rgba(0,229,255,0.65);
    animation: ai-pulse 1.4s infinite;
  }
  @keyframes ai-pulse{
    0%{ box-shadow: 0 0 0 0 rgba(0,229,255,0.55); }
    70%{ box-shadow: 0 0 0 10px rgba(0,229,255,0); }
    100%{ box-shadow: 0 0 0 0 rgba(0,229,255,0); }
  }

  /* Hero / ECG strip */
  .ai-hero{
    margin: 0.2rem auto 1.2rem auto;
    max-width: 1150px;
    padding: 1.2rem 1.2rem 0.2rem 1.2rem;
  }
  .ai-kicker{
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(0,229,255,0.65);
    margin-bottom: 0.6rem;
  }
  .ai-title{
    font-size: 2.8rem;
    line-height: 1.04;
    font-weight: 800;
    letter-spacing: -0.9px;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(135deg, #ffffff 0%, #bff9ff 35%, rgba(0,229,255,0.9) 70%, rgba(124,58,237,0.9) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .ai-sub{
    color: rgba(154,167,182,0.95);
    font-size: 0.98rem;
    max-width: 80ch;
  }
  .ai-ecg{
    margin: 1.15rem 0 0.25rem 0;
    border-radius: 18px;
    padding: 0.85rem 1rem;
    background: linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,229,255,0.14);
    box-shadow: 0 16px 50px rgba(0,0,0,0.55);
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 1rem;
  }
  .ai-ecg-left{
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    color: rgba(154,167,182,0.9);
    text-transform: uppercase;
    white-space: nowrap;
  }
  .ai-ecg-right{
    font-size: 0.9rem;
    font-weight: 700;
    color: rgba(255, 67, 111, 0.95);
    text-shadow: 0 0 18px rgba(255, 67, 111, 0.22);
    white-space: nowrap;
  }
  .ai-ecg-line{
    flex: 1;
    height: 34px;
    position: relative;
    overflow: hidden;
  }
  .ai-ecg-line svg{
    width: 140%;
    height: 34px;
    position:absolute;
    left:-20%;
    top:0;
    animation: ai-slide 2.6s linear infinite;
    filter: drop-shadow(0 0 12px rgba(0,229,255,0.28));
  }
  @keyframes ai-slide{
    0%{ transform: translateX(-10%); opacity: .95; }
    100%{ transform: translateX(10%); opacity: .95; }
  }

  /* Sidebar brand + user block */
  .ai-brand{
    margin: 0.25rem 0 0.9rem 0;
    padding: 0.75rem 0.75rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border: 1px solid rgba(0,229,255,0.14);
  }
  .ai-brand-row{
    display:flex; align-items:center; gap: .6rem;
  }
  .ai-brand-name{
    font-weight: 800;
    letter-spacing: 0.10em;
    color: rgba(0,229,255,0.95);
  }
  .ai-brand-sub{
    margin-top: 0.25rem;
    font-size: 0.72rem;
    color: rgba(154,167,182,0.85);
  }
  .ai-user{
    display:flex; align-items:center; gap:.75rem;
    padding: 0.85rem 0.75rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02));
    border: 1px solid rgba(124,58,237,0.16);
  }
  .ai-avatar{
    width: 38px; height: 38px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(0,229,255,0.9), rgba(124,58,237,0.9));
    display:flex; align-items:center; justify-content:center;
    color: #051018;
    font-weight: 800;
    box-shadow: 0 10px 30px rgba(0,229,255,0.12);
  }
  .ai-user-name{ font-weight: 700; color: rgba(232,238,249,0.95); }
  .ai-user-email{ font-size: .74rem; color: rgba(154,167,182,0.9); margin-top: 2px; }
  .ai-nav-title{
    margin: 1rem 0 0.3rem 0;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(154,167,182,0.8);
  }
</style>
    """,
    unsafe_allow_html=True,
)

def render_topbar(page_label: str):
    crumb = f"// DASHBOARD / {page_label.upper()}"
    st.markdown(
        f"""
        <div class="ai-topbar">
          <div class="ai-crumb">{crumb}</div>
          <div class="ai-status"><span class="ai-dot"></span> SYSTEM ONLINE — ALL SENSORS ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Persistence Helpers ─────────────────────────────────────────────────────
def load_stress_log_from_csv():
    """Load persisted stress history into session state (if available)."""
    if "stress_log" in st.session_state and st.session_state.stress_log:
        # Already populated in this session; don't overwrite
        return
    if os.path.exists(STRESS_LOG_PATH):
        try:
            df = pd.read_csv(STRESS_LOG_PATH, parse_dates=["timestamp"])
            records = df.to_dict("records")
            st.session_state.stress_log = records
        except Exception:
            # Fallback to empty log on any parsing error
            st.session_state.stress_log = []


def save_stress_log_to_csv():
    """Persist the current in-memory stress log to a CSV file."""
    try:
        if not st.session_state.get("stress_log"):
            # If log is empty, still write an empty file with columns for consistency
            empty_cols = ["timestamp", "hr", "rmssd", "scl", "stress_prob", "prediction"]
            pd.DataFrame(columns=empty_cols).to_csv(STRESS_LOG_PATH, index=False)
            return
        df = pd.DataFrame(st.session_state.stress_log)
        df.to_csv(STRESS_LOG_PATH, index=False)
    except Exception:
        # Avoid crashing the app if disk write fails
        pass


def _get_stress_db_connection():
    """Create (or open) the SQLite database used for real-time stress history."""
    conn = sqlite3.connect(STRESS_DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stress_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            heart_rate REAL,
            rmssd REAL,
            respiration REAL,
            stress_level TEXT NOT NULL
        )
        """
    )
    return conn


def log_stress_prediction_to_db(ts, hr, rmssd, respiration, stress_level):
    """Insert a single stress prediction row into SQLite."""
    try:
        conn = _get_stress_db_connection()
        conn.execute(
            """
            INSERT INTO stress_history (timestamp, heart_rate, rmssd, respiration, stress_level)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat() if isinstance(ts, datetime) else str(ts),
                float(hr) if hr is not None else None,
                float(rmssd) if rmssd is not None else None,
                float(respiration) if respiration is not None else None,
                str(stress_level),
            ),
        )
        conn.commit()
    except Exception:
        # Database errors should never crash the UI
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def load_last_hour_stress_history():
    """Return a DataFrame with the last 1 hour of stress predictions from SQLite."""
    try:
        conn = _get_stress_db_connection()
        df = pd.read_sql_query(
            """
            SELECT timestamp, heart_rate, rmssd, respiration, stress_level
            FROM stress_history
            ORDER BY timestamp
            """,
            conn,
        )
        conn.close()
    except Exception:
        return pd.DataFrame(
            columns=["timestamp", "heart_rate", "rmssd", "respiration", "stress_level"]
        )

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return df

    latest_ts = df["timestamp"].max()
    cutoff = latest_ts - timedelta(hours=1)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


# ─── Authentication Functions ────────────────────────────────────────────────
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed


def _get_current_user_profile():
    """
    Return (username, profile_dict) for the currently logged-in user.
    Profile is stored in st.session_state.users and extended with Fitbit metadata.
    """
    username = st.session_state.get("username")
    users = st.session_state.get("users", {})
    if username and username in users:
        return username, users[username]
    return None, None


def _set_user_fitbit_tokens(access_token, refresh_token, expires_at):
    """
    Persist Fitbit tokens for the current user in session_state only.
    Tokens are not logged or exposed to the frontend.
    """
    _, profile = _get_current_user_profile()
    if profile is None:
        return
    fitbit_meta = profile.setdefault("fitbit", {})
    fitbit_meta["access_token"] = access_token
    fitbit_meta["refresh_token"] = refresh_token
    # Store expiry as ISO string for easy serialization
    fitbit_meta["expires_at"] = (
        expires_at.isoformat() if isinstance(expires_at, datetime) else None
    )

    # Maintain legacy session keys for existing logic
    st.session_state.fitbit_token = access_token
    st.session_state.fitbit_refresh = refresh_token
    st.session_state.fitbit_expires_at = expires_at


def _get_user_fitbit_tokens():
    """
    Load Fitbit tokens for the current user from session_state.
    Returns (access_token, refresh_token, expires_at_datetime_or_None).
    """
    _, profile = _get_current_user_profile()
    access_token = None
    refresh_token = None
    expires_at = None
    if profile is not None:
        fitbit_meta = profile.get("fitbit", {})
        access_token = fitbit_meta.get("access_token")
        refresh_token = fitbit_meta.get("refresh_token")
        expires_at_raw = fitbit_meta.get("expires_at")
        if isinstance(expires_at_raw, datetime):
            expires_at = expires_at_raw
        elif isinstance(expires_at_raw, str):
            try:
                expires_at = datetime.fromisoformat(expires_at_raw)
            except Exception:
                expires_at = None

    # Prefer explicit user-scoped tokens; fall back to legacy keys if needed
    access_token = st.session_state.get("fitbit_token", access_token)
    refresh_token = st.session_state.get("fitbit_refresh", refresh_token)
    expires_at = st.session_state.get("fitbit_expires_at", expires_at)
    return access_token, refresh_token, expires_at


def _clear_user_fitbit_tokens():
    """Remove Fitbit tokens for the current user from session_state."""
    username, profile = _get_current_user_profile()
    if profile is not None and "fitbit" in profile:
        profile["fitbit"] = {}
    st.session_state.fitbit_token = None
    st.session_state.fitbit_refresh = None
    st.session_state.fitbit_expires_at = None


def _ensure_fitbit_access_token():
    """
    Ensure we have a valid Fitbit access token for the current user.
    If expired and a refresh token is available, refresh it automatically.
    Returns access_token or None if unavailable.
    """
    try:
        import requests  # Local import to avoid hard dependency at module import time
    except ImportError:
        st.warning("Install `requests` for Fitbit integration: `pip install requests`")
        return None

    fitbit_client_id = _get_secret_or_env("FITBIT_CLIENT_ID")
    fitbit_client_secret = _get_secret_or_env("FITBIT_CLIENT_SECRET")
    if not fitbit_client_id or not fitbit_client_secret:
        return None

    access_token, refresh_token, expires_at = _get_user_fitbit_tokens()
    now = datetime.utcnow()

    # If token is present and not close to expiry, reuse it
    if access_token and isinstance(expires_at, datetime) and expires_at > now + timedelta(seconds=30):
        return access_token

    # If we have no refresh token, we cannot refresh silently
    if not refresh_token:
        return access_token

    token_url = "https://api.fitbit.com/oauth2/token"
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    auth_header = (fitbit_client_id + ":" + fitbit_client_secret).encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(auth_header).decode("utf-8"),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    try:
        resp = requests.post(token_url, data=data, headers=headers, timeout=10)
    except Exception:
        st.warning("Could not contact Fitbit to refresh your session. Please try reconnecting your device.")
        return access_token

    if resp.status_code != 200:
        # Do not log response body to avoid leaking tokens
        st.warning("Your Fitbit session has expired. Please reconnect your device.")
        return access_token

    tok = resp.json()
    new_access = tok.get("access_token") or access_token
    new_refresh = tok.get("refresh_token") or refresh_token
    expires_in = tok.get("expires_in")
    new_expires_at = (
        datetime.utcnow() + timedelta(seconds=expires_in)
        if isinstance(expires_in, (int, float))
        else None
    )
    _set_user_fitbit_tokens(new_access, new_refresh, new_expires_at)
    return new_access


def _revoke_and_disconnect_fitbit():
    """
    Revoke Fitbit tokens (best-effort) and clear them from the current session.
    Does not expose client secret or tokens in logs.
    """
    try:
        import requests  # Local import
    except ImportError:
        _clear_user_fitbit_tokens()
        return

    fitbit_client_id = _get_secret_or_env("FITBIT_CLIENT_ID")
    fitbit_client_secret = _get_secret_or_env("FITBIT_CLIENT_SECRET")
    if not fitbit_client_id or not fitbit_client_secret:
        _clear_user_fitbit_tokens()
        return

    access_token, refresh_token, _ = _get_user_fitbit_tokens()
    token_to_revoke = access_token or refresh_token
    if not token_to_revoke:
        _clear_user_fitbit_tokens()
        return

    revoke_url = "https://api.fitbit.com/oauth2/revoke"
    auth_header = (fitbit_client_id + ":" + fitbit_client_secret).encode("utf-8")
    headers = {
        "Authorization": "Basic " + base64.b64encode(auth_header).decode("utf-8"),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"token": token_to_revoke}

    try:
        # Best-effort: we don't care about specific response body here
        requests.post(revoke_url, data=data, headers=headers, timeout=10)
    except Exception:
        pass
    finally:
        _clear_user_fitbit_tokens()

# Default users database (in production, use a real database)
DEFAULT_USERS = {
    "admin": {
        "password": hash_password("admin123"),
        "name": "Administrator",
        "email": "admin@stressdetection.ai"
    },
    "user": {
        "password": hash_password("user123"),
        "name": "Test User",
        "email": "user@stressdetection.ai"
    },
    "demo": {
        "password": hash_password("demo123"),
        "name": "Demo User",
        "email": "demo@stressdetection.ai"
    }
}

def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "users" not in st.session_state:
        st.session_state.users = DEFAULT_USERS.copy()
    if "stress_log" not in st.session_state:
        st.session_state.stress_log = []
        # Attempt to restore any previously saved history
        load_stress_log_from_csv()
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = None
    # Real-time monitoring state
    if "rt_monitoring" not in st.session_state:
        st.session_state.rt_monitoring = False
    if "rt_source" not in st.session_state:
        st.session_state.rt_source = "Fitbit API"
    if "rt_buffer" not in st.session_state:
        # List of records: {"timestamp", "hr", "rmssd", "respiration", "temp", "movement"}
        st.session_state.rt_buffer = []
    if "rt_last_window_end" not in st.session_state:
        st.session_state.rt_last_window_end = None
    if "rt_csv_df" not in st.session_state:
        st.session_state.rt_csv_df = None
    if "rt_csv_cursor" not in st.session_state:
        st.session_state.rt_csv_cursor = None

def generate_auth_token(username):
    """Generate a simple auth token"""
    return hashlib.sha256(f"{username}_stress_detection_secret".encode()).hexdigest()

def verify_auth_token(token, username):
    """Verify auth token"""
    expected_token = generate_auth_token(username)
    return token == expected_token

def login_user(username, password):
    """Authenticate user"""
    if username in st.session_state.users:
        if verify_password(password, st.session_state.users[username]["password"]):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.auth_token = generate_auth_token(username)
            return True
    return False

def logout_user():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.auth_token = None
    st.session_state.stress_log = []  # Clear stress log on logout

def check_auth_from_query():
    """Check authentication from query parameters"""
    try:
        # Try to get auth info from query params
        query_params = st.query_params
        if "auth" in query_params and "user" in query_params:
            username = query_params["user"]
            token = query_params["auth"]
            if username in st.session_state.users:
                if verify_auth_token(token, username):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.auth_token = token
                    return True
    except:
        pass
    return False

def set_auth_query_params():
    """Set authentication in query parameters"""
    if st.session_state.authenticated and st.session_state.username:
        try:
            st.query_params["auth"] = st.session_state.auth_token
            st.query_params["user"] = st.session_state.username
        except:
            pass

# Initialize session state FIRST
init_session_state()

# Check if we can restore auth from query params
if not st.session_state.authenticated:
    check_auth_from_query()

# Set query params if authenticated
if st.session_state.authenticated:
    set_auth_query_params()

# ─── Login Page ─────────────────────────────────────────────────────────────
def show_login_page():
    """Display login page"""
    # Hide sidebar, header, and deploy button for login page
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none !important;
    }
    .main .block-container {
        padding-top: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 100% !important;
    }
    .stApp > header {
        display: none;
    }
    /* Hide Streamlit deploy button */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    /* Hide deploy button */
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use columns to center the login box
    col_left, col_center, col_right = st.columns([1, 1.2, 1])
    
    with col_center:
        # Login box container
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1a1f2e 0%, #252d3d 100%);
            border: 1px solid rgba(0, 212, 170, 0.2);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            margin: 2rem auto;
        ">
        """, unsafe_allow_html=True)
        
        # Title
        st.markdown("""
        <h1 style="
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7b68ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            line-height: 1.2;
        ">🧠 Stress Detection</h1>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <p style="
            text-align: center;
            color: #8892a4;
            font-size: 1rem;
            margin-bottom: 2rem;
        ">Sign in to access the dashboard</p>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "👤 Username", 
                placeholder="Enter your username", 
                key="login_username"
            )
            password = st.text_input(
                "🔒 Access Code",
                placeholder="Enter your access code (not stored)",
                key="login_password"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns([1, 1], gap="small")
            with col_btn1:
                login_button = st.form_submit_button("🔐 Login", use_container_width=True, type="primary")
            with col_btn2:
                register_button = st.form_submit_button("📝 Register", use_container_width=True)
            
            if login_button:
                if username and password:
                    if login_user(username, password):
                        # Set query params for persistent auth
                        set_auth_query_params()
                        st.success(f"✅ Welcome back, {st.session_state.users[username]['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
                else:
                    st.warning("⚠️ Please enter both username and password.")
            
            if register_button:
                st.info("💡 Registration feature coming soon!")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ─── Check Authentication ────────────────────────────────────────────────────
if not st.session_state.authenticated:
    show_login_page()
    st.stop()  # Stop execution if not authenticated

# ─── Force Sidebar Visibility After Authentication ─────────────────────────
# Add CSS to ensure sidebar is visible (override any login page CSS)
st.markdown("""
<style>
    /* Force sidebar to be visible when authenticated - highest priority */
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    div[data-testid="stSidebar"],
    .css-1d391kg[data-testid="stSidebar"],
    .css-1lcbmhc[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        transform: translateX(0) !important;
        width: 21rem !important;
        min-width: 21rem !important;
    }
    /* Ensure sidebar content is visible */
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > * {
        display: block !important;
        visibility: visible !important;
    }
    /* Show sidebar toggle button */
    [data-testid="collapsedControl"],
    button[data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
    }
</style>
<script>
    // Ensure sidebar is visible via JavaScript as well
    setTimeout(function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
            sidebar.style.opacity = '1';
            sidebar.style.transform = 'translateX(0)';
        }
    }, 100);
</script>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="ai-brand">
          <div class="ai-brand-row">
            <div style="width:28px;height:28px;border-radius:10px;background:rgba(0,229,255,0.10);
                        display:flex;align-items:center;justify-content:center;border:1px solid rgba(0,229,255,0.22);">
              <span style="color:rgba(0,229,255,0.95);font-weight:900;">∿</span>
            </div>
            <div class="ai-brand-name">STRESSAI</div>
          </div>
          <div class="ai-brand-sub">// stress-detection-v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # User info and logout
    if st.session_state.authenticated:
        user_info = st.session_state.users.get(st.session_state.username, {})
        initials = (user_info.get("name") or st.session_state.username or "U").strip()[:2].upper()
        st.markdown(
            f"""
            <div class="ai-user">
              <div class="ai-avatar">{initials}</div>
              <div>
                <div class="ai-user-name">{user_info.get('name', st.session_state.username)}</div>
                <div class="ai-user-email">{user_info.get('email','')}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            logout_user()
            # Clear query params on logout
            try:
                st.query_params.clear()
            except:
                pass
            st.rerun()
    
    st.markdown("")
    st.markdown('<div class="ai-nav-title">// NAVIGATION</div>', unsafe_allow_html=True)

    nav_options = [
        "🏠 Home",
        "🔮 Predict Stress",
        "⌚ Smartwatch",
        "🔄 Long-Term Analytics",
        "📊 Model Results",
        "📈 Data Explorer",
        "📖 About",
    ]

    # When returning from Fitbit callback, prefer to show the Smartwatch page once
    default_index = 0
    try:
        if st.query_params.get("code"):
            default_index = nav_options.index("⌚ Smartwatch")
    except Exception:
        default_index = 0

    page = st.radio(
        "Navigate",
        nav_options,
        index=default_index,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#8892a4; font-size:0.75rem; text-align:center;'>"
        "Personalized ML for<br>Stress Detection<br>"
        "<span style='color:#00d4aa;'>v1.0</span></p>",
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    render_topbar("Home")
    st.markdown(
        """
        <div class="ai-hero">
          <div class="ai-kicker">// PERSONALIZED MACHINE LEARNING</div>
          <div class="ai-title">Stress Detection<br/>via Multimodal<br/>Physiological Data</div>
          <div class="ai-sub">
            Continuous biometric monitoring, feature extraction, and predictive analytics—designed as a
            clean, futuristic dashboard.
          </div>
          <div class="ai-ecg">
            <div class="ai-ecg-left">LIVE ECG SIGNAL</div>
            <div class="ai-ecg-line">
              <svg viewBox="0 0 1200 34" preserveAspectRatio="none">
                <path d="M0 17 L120 17 L150 17 L170 4 L190 30 L210 17 L310 17 L340 17 L360 8 L380 27 L400 17 L520 17
                         L560 17 L585 10 L600 17 L615 24 L630 17 L760 17 L790 17 L810 5 L830 30 L850 17 L980 17
                         L1010 17 L1030 9 L1050 26 L1070 17 L1200 17"
                      fill="none" stroke="rgba(0,229,255,0.95)" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
              </svg>
            </div>
            <div class="ai-ecg-right">92 BPM</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">0.9960</div>'
            '<div class="metric-label">Best F1 (Lab)</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">0.8131</div>'
            '<div class="metric-label">Best F1 (Wild)</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">4</div>'
            '<div class="metric-label">Datasets</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">5</div>'
            '<div class="metric-label">ML Approaches</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Abstract
    st.markdown('<div class="section-header">📄 Abstract</div>', unsafe_allow_html=True)
    st.markdown("""
    This work investigates the feasibility of **stress detection** using **physiological measurements** 
    captured by smartwatches and personalized machine learning techniques. We group users based on 
    stress-indicative attributes (exercising, sleep quality, personality) and train personalized ML models 
    for each group or use **multitask learning (MTL)**.

    The best results are achieved using **multi-attribute-based models**, with up to **0.9960 F1-score** 
    in lab settings and **0.8131 F1-score** in-the-wild datasets.
    """)

    # Datasets overview
    st.markdown('<div class="section-header">📂 Datasets</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            '<div class="metric-card">'
            '<span class="dataset-badge badge-lab">LAB</span><br><br>'
            '<span style="font-size:1.3rem; font-weight:700;">WESAD</span><br>'
            '<span style="color:#8892a4; font-size:0.85rem;">'
            '15 participants · Empatica E4<br>HR, SCL, Temp, Acceleration</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-card">'
            '<span class="dataset-badge badge-wild">IN-THE-WILD</span><br><br>'
            '<span style="font-size:1.3rem; font-weight:700;">ADARP</span><br>'
            '<span style="color:#8892a4; font-size:0.85rem;">'
            '11 participants · 14 days<br>HR, SCL, Temp, Acceleration</span></div>',
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            '<div class="metric-card">'
            '<span class="dataset-badge badge-lab">LAB</span><br><br>'
            '<span style="font-size:1.3rem; font-weight:700;">SWELL-KW</span><br>'
            '<span style="color:#8892a4; font-size:0.85rem;">'
            '25 participants · Mobi device<br>HR, HRV, SCL</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            '<div class="metric-card">'
            '<span class="dataset-badge badge-wild">IN-THE-WILD</span><br><br>'
            '<span style="font-size:1.3rem; font-weight:700;">LifeSnaps</span><br>'
            '<span style="color:#8892a4; font-size:0.85rem;">'
            '71 participants · 4 months<br>Fitbit Sense · 60+ features</span></div>',
            unsafe_allow_html=True,
        )

    # Key Findings
    st.markdown('<div class="section-header">🔑 Key Findings</div>', unsafe_allow_html=True)

    findings = [
        ("🎯", "Personalization Matters",
         "User-based models consistently outperform generic ones."),
        ("🏆", "Multi-Attribute Models Win",
         "Grouping users by multiple attributes yields the best F1-scores across all datasets."),
        ("🤖", "MTL Boosts Deep Learning",
         "Multitask learning improved F1-Score by 0.88–158.7% over single-task learning."),
        ("🌍", "Lab vs. Wild Gap",
         "Lab settings achieve higher accuracy, highlighting real-world challenges."),
    ]

    for emoji, title, desc in findings:
        st.markdown(
            f'<div class="about-card">'
            f'<span style="font-size:1.3rem;">{emoji}</span> '
            f'<strong>{title}</strong><br>'
            f'<span style="color:#8892a4;">{desc}</span></div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT STRESS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Stress":
    render_topbar("Predict Stress")

    # Check if model exists
    model_path = os.path.join(MODEL_DIR, "swell_rf_model.pkl")
    if not os.path.exists(model_path):
        st.warning(
            "No trained model found. Run `python train_model.py` first to train the model."
        )
        st.code("python train_model.py", language="bash")
        st.stop()

    @st.cache_resource
    def load_model():
        model_path_local = os.path.join(MODEL_DIR, "swell_rf_model.pkl")
        scaler_path_local = os.path.join(MODEL_DIR, "swell_scaler.pkl")
        model = joblib.load(model_path_local)
        scaler = joblib.load(scaler_path_local)
        metadata = joblib.load(os.path.join(MODEL_DIR, "swell_metadata.pkl"))
        features_path_local = os.path.join(MODEL_DIR, "swell_features.pkl")
        features = ["HR", "RMSSD", "SCL"]
        try:
            if os.path.exists(features_path_local):
                features = joblib.load(features_path_local)
        except Exception:
            features = ["HR", "RMSSD", "SCL"]
        return model, scaler, metadata, features

    model, scaler, metadata, feature_cols = load_model()

    def _make_feature_vector(hr_v: float, rmssd_v: float, scl_v: float, cols):
        base = {
            "HR": float(hr_v),
            "RMSSD": float(rmssd_v),
            "SCL": float(scl_v),
            "HR_RMSSD_ratio": float(hr_v) / (float(rmssd_v) + 1.0),
            "Sympathetic_Index": float(hr_v) * float(scl_v),
            "HR_log": float(np.log1p(float(hr_v))),
            "RMSSD_log": float(np.log1p(float(rmssd_v))),
            "SCL_log": float(np.log1p(float(scl_v))),
        }
        vec = np.array([[base.get(c, 0.0) for c in cols]], dtype=float)
        expected = None
        try:
            expected = int(getattr(scaler, "n_features_in_", vec.shape[1]))
        except Exception:
            expected = vec.shape[1]
        if expected is not None and vec.shape[1] != expected:
            vec = vec[:, :expected] if vec.shape[1] > expected else np.pad(vec, ((0, 0), (0, expected - vec.shape[1])), constant_values=0.0)
        return vec

    # New UI: page header
    st.markdown("""
    <div class="pg-label">// real-time inference</div>
    <div class="pg-title">Predict Stress</div>
    <div class="pg-sub">Enter your physiological readings to get a personalised stress prediction</div>
    """, unsafe_allow_html=True)

    f1_display = (
        metadata.get("f1_score")
        or metadata.get("selected_metrics", {}).get("f1")
        or metadata.get("selected_metrics", {}).get("f1_score")
        or "—"
    )
    n_samples = metadata.get("n_samples", "—")
    model_type = metadata.get("model_type", "XGBClassifier")

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card c1">
        <div class="stat-lbl">Active Model</div>
        <div class="stat-val" style="font-size:18px">{model_type}</div>
        <div class="stat-meta">engineered features</div>
      </div>
      <div class="stat-card c2">
        <div class="stat-lbl">F1-Score</div>
        <div class="stat-val">{f1_display}</div>
        <div class="stat-meta">on training set</div>
      </div>
      <div class="stat-card c3">
        <div class="stat-lbl">Training Samples</div>
        <div class="stat-val">{n_samples}</div>
        <div class="stat-meta">labelled records</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="medium")

    with left_col:
        st.markdown("""
        <div class="sec-card">
          <div class="sec-head">
            <div class="sec-icon">❤️</div>
            <div><div class="sec-title">Physiological Signals</div><div class="sec-sub">Wearable sensor readings</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        hr = st.slider("Heart Rate (bpm) — Normal: 60–100", min_value=40, max_value=200, value=75, step=1, key="hr_slider")
        rmssd = st.slider("HRV · RMSSD (ms) — Higher = relaxed", min_value=5, max_value=200, value=42, step=1, key="rmssd_slider")
        scl = st.number_input("SCL (µS)", min_value=0.0, max_value=30.0, value=5.2, step=0.1, key="scl_input")
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  Predict Stress Level", type="primary", use_container_width=True)

    with right_col:
        if predict_btn:
            input_data = _make_feature_vector(hr, rmssd, scl, feature_cols)
            input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
            input_scaled = scaler.transform(input_data)
            probabilities = model.predict_proba(input_scaled)[0]
            prob_stress = float(probabilities[1])
            stress_prob = prob_stress * 100.0
            if prob_stress >= 0.60:
                stress_level = "high"
                emoji = "😰"
                label_text = "HIGH STRESS"
                color, glow = "#f87171", "rgba(248,113,113,0.1)"
            elif prob_stress >= 0.20:
                stress_level = "moderate"
                emoji = "😐"
                label_text = "MODERATE"
                color, glow = "#fbbf24", "rgba(251,191,36,0.1)"
            else:
                stress_level = "low"
                emoji = "😊"
                label_text = "LOW STRESS"
                color, glow = "#4ade80", "rgba(74,222,128,0.1)"
            bar_w = stress_prob
            st.session_state.stress_log.append({
                "timestamp": datetime.now(),
                "hr": hr,
                "rmssd": rmssd,
                "scl": scl,
                "stress_prob": stress_prob,
                "prediction": 2 if stress_level == "high" else 1 if stress_level == "moderate" else 0,
            })
            save_stress_log_to_csv()

            st.markdown(f"""
            <div class="result-box active" style="--rc:{color};--rg:{glow}">
              <div class="res-emoji">{emoji}</div>
              <div class="res-score" style="color:{color};text-shadow:0 0 30px {color}">{int(round(stress_prob))}</div>
              <div class="res-level" style="color:{color}">{label_text}</div>
              <div class="res-bar-wrap">
                <div class="res-bar-track">
                  <div class="res-bar-fill" style="width:{bar_w}%;background:linear-gradient(90deg,{color}88,{color});box-shadow:0 0 10px {color}"></div>
                </div>
              </div>
              <div style="font-size:12px; color:#8ba3c7; margin-top:10px;">Stress probability: {stress_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="bd-card">
              <div class="bd-title">Signal Breakdown</div>
              <div class="bd-row"><span class="bd-name">Heart Rate</span> <span class="bd-val">{hr} bpm</span></div>
              <div class="bd-row"><span class="bd-name">HRV · RMSSD</span> <span class="bd-val">{rmssd} ms</span></div>
              <div class="bd-row"><span class="bd-name">SCL</span> <span class="bd-val">{scl} µS</span></div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("🩺 Recommendations", expanded=False):
                if stress_level == "high":
                    st.markdown("**Stress reduction:** Try 4-7-8 breathing, a short walk, and avoid caffeine. Consider consulting a healthcare professional if stress is persistent.")
                elif stress_level == "moderate":
                    st.markdown("**Moderate stress:** Take 5-minute breaks, practice deep breathing 2–3 times today, stay hydrated, and limit caffeine.")
                else:
                    st.markdown("**Low stress:** Keep up good habits: consistent sleep, regular activity, and mindfulness.")
        else:
            st.markdown("""
            <div class="result-box">
              <div class="res-empty-icon">🔮</div>
              <div class="res-empty-text">Enter your values and<br>click "Predict Stress Level"</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ref-card">
          <div class="ref-head">📋 Reference Ranges</div>
          <div class="ref-body">
            <div class="ref-row"><span class="ref-sig">Heart Rate</span> <span class="ref-rng cy">60–100 bpm</span></div>
            <div class="ref-row"><span class="ref-sig">HRV (RMSSD)</span> <span class="ref-rng am">20–60 ms</span></div>
            <div class="ref-row"><span class="ref-sig">SCL</span> <span class="ref-rng gr">2–20 µS</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SMARTWATCH
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⌚ Smartwatch":
    render_topbar("Smartwatch")
    st.markdown('<h1 class="main-title">Smartwatch Integration</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Import HR, HRV, and stress data from Fitbit, Garmin, Apple Health, or any CSV export</p>',
        unsafe_allow_html=True,
    )

    # Check if model exists for smartwatch features
    model_path = os.path.join(MODEL_DIR, "swell_rf_model.pkl")
    if not os.path.exists(model_path):
        st.warning(
            "⚠️ No trained model found. Run `python train_model.py` first to train the model."
        )
        st.stop()

    st.info("✅ **Import from CSV** works immediately — no setup required. Use it for Fitbit, Garmin, Apple Health, or any device export.")
    st.markdown("")

    tab_import, tab_fitbit, tab_realtime = st.tabs(
        [
            "📤 Import from CSV (recommended)",
            "⌚ Connect Fitbit (API)",
            "📡 Real-Time Monitoring",
        ]
    )

    with tab_import:
        st.markdown(
            '<div class="section-header">📤 Import Data from Smartwatch Export</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#8892a4;">Upload a CSV file exported from Fitbit, Garmin, Apple Health (via export apps), '
            'or any device. We need <strong>HR</strong> (heart rate). <strong>RMSSD</strong> (heart rate variability) is optional '
            'and will be estimated from HR if missing. SCL (skin conductance) is optional — we will estimate it from '
            'HR and RMSSD if not present.</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Export your data from Fitbit (Data Export), Garmin Connect, or Apple Health export apps.",
        )
        if uploaded_file is not None:
            try:
                df_import = pd.read_csv(uploaded_file)
                st.dataframe(df_import.head(10), use_container_width=True, height=200)
                st.caption(f"Columns: {list(df_import.columns)}")
                numeric_cols = df_import.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found. Ensure your CSV has HR, RMSSD, or similar columns.")
                else:
                    hr_cols = [c for c in numeric_cols if any(x in c.lower() for x in ["hr", "heart", "bpm", "rate"])]
                    rmssd_cols = [c for c in numeric_cols if any(x in c.lower() for x in ["rmssd", "hrv", "variability"])]
                    scl_cols = [c for c in numeric_cols if any(x in c.lower() for x in ["scl", "sc", "conductance", "eda", "electrodermal"])]
                    col_hr, col_rmssd, col_scl = st.columns(3)
                    with col_hr:
                        hr_idx = numeric_cols.index(hr_cols[0]) if hr_cols and hr_cols[0] in numeric_cols else 0
                        hr_col = st.selectbox("HR (heart rate) column", options=numeric_cols, index=hr_idx)
                    with col_rmssd:
                        rmssd_opts = ["(none)"] + numeric_cols
                        rmssd_idx = rmssd_opts.index(rmssd_cols[0]) if rmssd_cols and rmssd_cols[0] in numeric_cols else 0
                        rmssd_col = st.selectbox("RMSSD (HRV) column", options=rmssd_opts, index=rmssd_idx)
                    with col_scl:
                        scl_opts = ["(none - estimate)"] + numeric_cols
                        scl_idx = scl_opts.index(scl_cols[0]) if scl_cols and scl_cols[0] in numeric_cols else 0
                        scl_col = st.selectbox("SCL (skin conductance) column", options=scl_opts, index=scl_idx)
                    if st.button("🔄 Import and Predict Stress"):
                        hr_vals = pd.to_numeric(df_import[hr_col], errors="coerce").dropna()
                        hr_use = float(hr_vals.iloc[-1]) if len(hr_vals) else 75

                        if rmssd_col == "(none)":
                            rmssd_use = estimate_rmssd_from_hr(hr_use)
                            st.info(f"RMSSD estimated from HR: {rmssd_use:.1f} ms (no HRV column provided)")
                        else:
                            rmssd_vals = pd.to_numeric(df_import[rmssd_col], errors="coerce").dropna()
                            rmssd_use = float(rmssd_vals.iloc[-1]) if len(rmssd_vals) else estimate_rmssd_from_hr(hr_use)
                            if not len(rmssd_vals):
                                st.info(f"RMSSD estimated from HR: {rmssd_use:.1f} ms (selected RMSSD column had no valid numeric values)")

                        if scl_col and scl_col != "(none - estimate)":
                            scl_vals = pd.to_numeric(df_import[scl_col], errors="coerce").dropna()
                            scl_use = float(scl_vals.iloc[-1]) if len(scl_vals) else estimate_scl_from_hr_rmssd(hr_use, rmssd_use)
                        else:
                            scl_use = estimate_scl_from_hr_rmssd(hr_use, rmssd_use)
                            st.info(f"SCL estimated from HR/RMSSD: {scl_use:.1f} µS (smartwatch typically does not provide SCL)")

                        model = joblib.load(model_path)
                        scaler = joblib.load(os.path.join(MODEL_DIR, "swell_scaler.pkl"))
                        metadata = joblib.load(os.path.join(MODEL_DIR, "swell_metadata.pkl"))
                        features = ["HR", "RMSSD", "SCL"]
                        try:
                            features = joblib.load(os.path.join(MODEL_DIR, "swell_features.pkl"))
                        except Exception:
                            features = ["HR", "RMSSD", "SCL"]
                        base = {
                            "HR": float(hr_use),
                            "RMSSD": float(rmssd_use),
                            "SCL": float(scl_use),
                            "HR_RMSSD_ratio": float(hr_use) / (float(rmssd_use) + 1.0),
                            "Sympathetic_Index": float(hr_use) * float(scl_use),
                            "HR_log": float(np.log1p(float(hr_use))),
                            "RMSSD_log": float(np.log1p(float(rmssd_use))),
                            "SCL_log": float(np.log1p(float(scl_use))),
                        }
                        input_data = np.array([[base.get(c, 0.0) for c in features]], dtype=float)
                        input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
                        input_scaled = scaler.transform(input_data)
                        probabilities = model.predict_proba(input_scaled)[0]
                        prob_stress = float(probabilities[1])
                        stress_prob = prob_stress * 100.0
                        if prob_stress >= 0.60:
                            pred_label, pred_val = "High Stress", 2
                        elif prob_stress >= 0.20:
                            pred_label, pred_val = "Moderate Stress", 1
                        else:
                            pred_label, pred_val = "No Stress", 0
                        st.success(f"**Imported:** HR={hr_use:.0f} bpm, RMSSD={rmssd_use:.1f} ms, SCL={scl_use:.1f} µS")
                        st.markdown(f'<div class="prediction-box stress-{"low" if pred_val==0 else "moderate" if pred_val==1 else "high"}">'
                                    f'<div class="pred-label">{pred_label}</div>'
                                    f'<div class="pred-confidence">Stress Probability: {stress_prob:.1f}%</div></div>',
                                    unsafe_allow_html=True)
                        st.session_state.stress_log.append({
                            "timestamp": datetime.now(), "hr": hr_use, "rmssd": rmssd_use,
                            "scl": scl_use, "stress_prob": stress_prob, "prediction": int(pred_val),
                        })
                        save_stress_log_to_csv()
                        st.balloons()
            except Exception as e:
                st.error(f"Error reading file: {e}")

    with tab_fitbit:
        # ── Section header ──
        st.markdown(
            '<div class="section-header">⌚ Connect Your Device</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#8892a4;">Connect your Fitbit account or bring in data from any smartwatch via CSV, '
            'or enter key metrics manually. Stress is always predicted with the same secure pipeline — only the '
            'input method changes.</p>',
            unsafe_allow_html=True,
        )

        # Local CSS for glassmorphism cards and responsive grid
        st.markdown(
            """
            <style>
            .device-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1.25rem;
                margin-top: 1.25rem;
            }
            @media (max-width: 900px) {
                .device-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 640px) {
                .device-grid {
                    grid-template-columns: minmax(0, 1fr);
                }
            }
            .device-card {
                position: relative;
                padding: 1.2rem 1.4rem;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,64,175,0.55));
                border: 1px solid rgba(148,163,184,0.4);
                box-shadow: 0 18px 45px rgba(15,23,42,0.65);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, border-color 0.18s ease-out;
            }
            .device-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 24px 60px rgba(15,23,42,0.75);
                border-color: rgba(129,140,248,0.85);
            }
            .device-title {
                font-size: 0.95rem;
                font-weight: 600;
                letter-spacing: 0.02em;
                color: #e5e7eb;
                margin-bottom: 0.35rem;
            }
            .device-subtitle {
                font-size: 0.8rem;
                color: #9ca3af;
                margin-bottom: 0.6rem;
            }
            .device-badge {
                display: inline-flex;
                align-items: center;
                padding: 0.18rem 0.55rem;
                border-radius: 999px;
                font-size: 0.68rem;
                background: rgba(15,118,110,0.12);
                color: #a5b4fc;
                border: 1px solid rgba(129,140,248,0.4);
                margin-bottom: 0.6rem;
            }
            .device-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.45rem 0.95rem;
                border-radius: 999px;
                border: none;
                outline: none;
                font-size: 0.8rem;
                font-weight: 500;
                color: #0b1020;
                background: linear-gradient(135deg,#f97316,#facc15);
                text-decoration: none;
                cursor: pointer;
                box-shadow: 0 8px 20px rgba(248,113,113,0.45);
            }
            .device-btn-secondary {
                background: rgba(15,23,42,0.7);
                color: #e5e7eb;
                border: 1px solid rgba(148,163,184,0.6);
                box-shadow: none;
            }
            .device-field-label {
                font-size: 0.75rem;
                color: #9ca3af;
                margin-bottom: 0.15rem;
            }
            .file-chip {
                display: inline-flex;
                align-items: center;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.7);
                font-size: 0.7rem;
                color: #e5e7eb;
                margin-top: 0.4rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # ── Existing Fitbit OAuth / API plumbing (unchanged semantics) ──
        try:
            import requests
        except ImportError:
            st.warning("Install `requests` for Fitbit integration: `pip install requests`")
            st.stop()

        fitbit_client_id = _get_secret_or_env("FITBIT_CLIENT_ID")
        fitbit_client_secret = _get_secret_or_env("FITBIT_CLIENT_SECRET")

        if not fitbit_client_id or not fitbit_client_secret:
            # No keys configured – keep original helper text
            st.success("💡 **No Fitbit API keys?** Use **Import from CSV** above — it works with Fitbit data exports and requires no setup.")
            st.markdown("---")
            st.markdown("**Optional: Fitbit OAuth (live API connection)**")
            st.markdown(
                "1. Go to [dev.fitbit.com](https://dev.fitbit.com/apps/new) and create an app.\n"
                "2. Set Callback URL to `http://localhost:8505/callback` (or your deployed app URL with `/callback`).\n"
                "3. Add credentials to `.streamlit/secrets.toml` or environment variables:"
            )
            st.code(
                "FITBIT_CLIENT_ID = \"your_id\"\n"
                "FITBIT_CLIENT_SECRET = \"your_secret\"",
                language="toml",
            )
        else:
            if "fitbit_token" not in st.session_state:
                st.session_state.fitbit_token = None

            # Handle OAuth callback: /callback?code=...
            auth_code = st.query_params.get("code")
            oauth_error = st.query_params.get("error")
            if oauth_error and not auth_code:
                st.error("Authorization failed. Please try connecting Fitbit again.")
            if auth_code and not st.session_state.fitbit_token:
                token_url = "https://api.fitbit.com/oauth2/token"
                redirect_uri = _get_secret_or_env("FITBIT_REDIRECT_URI", "http://localhost:8505/callback")
                data = {
                    "client_id": fitbit_client_id,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                    "code": auth_code,
                }
                auth_header = (fitbit_client_id + ":" + fitbit_client_secret).encode("utf-8")
                headers = {"Authorization": "Basic " + base64.b64encode(auth_header).decode("utf-8")}
                try:
                    resp = requests.post(token_url, data=data, headers=headers, timeout=10)
                except Exception:
                    st.error("Unable to contact Fitbit for token exchange. Please try again.")
                else:
                    if resp.status_code == 200:
                        tok = resp.json()
                        access_token = tok.get("access_token")
                        refresh_token = tok.get("refresh_token")
                        expires_in = tok.get("expires_in")
                        expires_at = (
                            datetime.utcnow() + timedelta(seconds=expires_in)
                            if isinstance(expires_in, (int, float))
                            else None
                        )
                        _set_user_fitbit_tokens(access_token, refresh_token, expires_at)
                        # Toast-style feedback where available
                        try:
                            st.toast("Fitbit connected successfully")
                        except Exception:
                            st.success("Fitbit connected successfully")
                        # Clear query params so code/error is not kept in the URL
                        try:
                            st.query_params.clear()
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error("Fitbit OAuth token exchange failed. Please try reconnecting your device.")

            access_token_existing, _, _ = _get_user_fitbit_tokens()
            has_fitbit_token = bool(access_token_existing)

            # Pre-fetch latest Fitbit HR / HRV when token is present (same behavior as before)
            hr_use = None
            rmssd_use = None
            scl_use = None

            if has_fitbit_token:
                access_token = _ensure_fitbit_access_token()
                if not access_token:
                    st.warning("Fitbit session is not available. Please reconnect your device.")
                else:
                    headers = {"Authorization": "Bearer " + access_token}
                today = datetime.now().strftime("%Y-%m-%d")
                try:
                    hr_resp = requests.get(
                        f"https://api.fitbit.com/1/user/-/activities/heart/date/{today}/1d.json",
                        headers=headers,
                    )
                    hrv_resp = requests.get(
                        f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json",
                        headers=headers,
                    )
                    hr_val, rmssd_val = None, None
                    if hr_resp.status_code == 200:
                        j = hr_resp.json()
                        ah = j.get("activities-heart", [])
                        if ah:
                            val = ah[0].get("value", {})
                            hr_val = val.get("restingHeartRate")
                            if hr_val is None and val.get("heartRateZones"):
                                for z in val["heartRateZones"]:
                                    hr_val = z.get("max") or z.get("min")
                                    if hr_val is not None:
                                        break
                        if hr_val is None and "activities-heart-intraday" in j:
                            hri = j.get("activities-heart-intraday", {})
                            ds = hri.get("dataset", [])
                            if ds:
                                hr_val = sum(d.get("value", 0) for d in ds[-10:]) / min(10, len(ds))
                    if hrv_resp.status_code == 200:
                        hrv = hrv_resp.json().get("hrv", [{}])
                        if hrv and hrv[0].get("value", {}).get("dailyRmssd"):
                            rmssd_val = hrv[0]["value"]["dailyRmssd"]
                    if hr_val is not None or rmssd_val is not None:
                        hr_use = hr_val if hr_val is not None else 75
                        rmssd_use = rmssd_val if rmssd_val is not None else 42
                        scl_use = estimate_scl_from_hr_rmssd(hr_use, rmssd_use)
                    else:
                        st.warning("No HR or HRV data available today. Sync your Fitbit and ensure HR/HRV tracking is enabled.")
                except Exception:
                    st.error("Unable to fetch Fitbit data. Please check your connection and try again.")

            # ── Three-card layout ──
            col1, col2, col3 = st.columns(3)

            # Card 1: Fitbit (Cloud API)
            with col1:
                st.markdown(
                    """
                    <div class="device-card">
                        <div class="device-badge">Fitbit API</div>
                        <div class="device-title">Fitbit (Cloud API)</div>
                        <div class="device-subtitle">Secure OAuth-based data sync directly from your Fitbit account.</div>
                    """,
                    unsafe_allow_html=True,
                )

                if has_fitbit_token:
                    st.markdown(
                        '<p style="font-size:0.8rem; color:#a5b4fc; margin-top:0.1rem;">'
                        'Connected to Fitbit. Fetch the latest data and predict your current stress.</p>',
                        unsafe_allow_html=True,
                    )
                    # Allow user to disconnect / revoke
                    if st.button("Disconnect Fitbit", key="fitbit_disconnect_btn", use_container_width=True):
                        _revoke_and_disconnect_fitbit()
                        st.success("Fitbit disconnected.")
                        st.rerun()
                    if hr_use is not None or rmssd_use is not None:
                        if st.button("🔄 Fetch Latest & Predict", key="fitbit_fetch_btn", use_container_width=True):
                            model = joblib.load(model_path)
                            scaler = joblib.load(os.path.join(MODEL_DIR, "swell_scaler.pkl"))
                            hr_v = hr_use if hr_use is not None else 75
                            rmssd_v = rmssd_use if rmssd_use is not None else 42
                            scl_v = scl_use if scl_use is not None else estimate_scl_from_hr_rmssd(75, 42)
                            features = ["HR", "RMSSD", "SCL"]
                            try:
                                features = joblib.load(os.path.join(MODEL_DIR, "swell_features.pkl"))
                            except Exception:
                                features = ["HR", "RMSSD", "SCL"]
                            base = {
                                "HR": float(hr_v),
                                "RMSSD": float(rmssd_v),
                                "SCL": float(scl_v),
                                "HR_RMSSD_ratio": float(hr_v) / (float(rmssd_v) + 1.0),
                                "Sympathetic_Index": float(hr_v) * float(scl_v),
                                "HR_log": float(np.log1p(float(hr_v))),
                                "RMSSD_log": float(np.log1p(float(rmssd_v))),
                                "SCL_log": float(np.log1p(float(scl_v))),
                            }
                            input_data = np.array([[base.get(c, 0.0) for c in features]], dtype=float)
                            input_data = np.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
                            input_scaled = scaler.transform(input_data)
                            probabilities = model.predict_proba(input_scaled)[0]
                            prob_stress = float(probabilities[1])
                            stress_prob = prob_stress * 100.0
                            if prob_stress >= 0.60:
                                pred_label, pred_val = "High Stress", 2
                            elif prob_stress >= 0.20:
                                pred_label, pred_val = "Moderate Stress", 1
                            else:
                                pred_label, pred_val = "No Stress", 0
                            st.success(
                                f"HR={hr_use:.0f} bpm, RMSSD={rmssd_use:.1f} ms (SCL estimated: {scl_use:.1f} µS)"
                            )
                            st.markdown(
                                f'<div class="prediction-box stress-{"low" if pred_val==0 else "moderate" if pred_val==1 else "high"}">'
                                f'<div class="pred-label">{pred_label}</div>'
                                f'<div class="pred-confidence">Stress Probability: {stress_prob:.1f}%</div></div>',
                                unsafe_allow_html=True,
                            )
                            st.session_state.stress_log.append(
                                {
                                    "timestamp": datetime.now(),
                                    "hr": hr_use,
                                    "rmssd": rmssd_use,
                                    "scl": scl_use,
                                    "stress_prob": stress_prob,
                                    "prediction": int(pred_val),
                                }
                            )
                            save_stress_log_to_csv()
                    else:
                        st.caption(
                            "Waiting for today’s Fitbit HR / HRV data. Open the Fitbit app and sync your watch."
                        )
                else:
                    redirect_uri = _get_secret_or_env("FITBIT_REDIRECT_URI", "http://localhost:8505/callback")
                    scope = "heartrate activity"
                    # Explicitly URL-encode the redirect URI
                    encoded_redirect = urllib.parse.quote(redirect_uri, safe="")
                    auth_url = (
                        "https://www.fitbit.com/oauth2/authorize"
                        f"?response_type=code"
                        f"&client_id={fitbit_client_id}"
                        f"&scope={scope}"
                        f"&redirect_uri={encoded_redirect}"
                    )
                    st.markdown(
                        f'<a href="{auth_url}" class="device-btn">Connect Fitbit</a>',
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "You will be redirected to Fitbit to authorize, then returned here automatically."
                    )

            # Card 2: Universal CSV Upload (UI only – reuses Import from CSV flow)
            with col2:
                st.markdown(
                    """
                    <div class="device-card">
                        <div class="device-badge">Any Smartwatch</div>
                        <div class="device-title">Universal CSV Upload</div>
                        <div class="device-subtitle">
                            Drag and drop CSV exports from Redmi, Garmin, Apple Health, or any device.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                csv_file = st.file_uploader(
                    "Drop CSV here",
                    type=["csv"],
                    key="device_csv_upload",
                    label_visibility="collapsed",
                )
                if csv_file is not None:
                    st.markdown(
                        f'<div class="file-chip">📄 {csv_file.name}</div>',
                        unsafe_allow_html=True,
                    )
                if st.button("Upload Data", key="device_csv_upload_btn", use_container_width=True):
                    if csv_file is None:
                        st.warning("Please select a CSV file first.")
                    else:
                        # UI hint – actual processing still happens via the dedicated Import from CSV tab
                        st.info(
                            "CSV selected. Use the **Import from CSV (recommended)** tab above to map columns "
                            "and run detailed stress predictions."
                        )

            # Card 3: Manual Entry (UI wrapper; uses existing Predict Stress page logic for full analysis)
            with col3:
                st.markdown(
                    """
                    <div class="device-card">
                        <div class="device-badge">Quick Check</div>
                        <div class="device-title">Manual Entry</div>
                        <div class="device-subtitle">
                            No wearable? Enter a few metrics for a quick stress snapshot.
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                hr_manual = st.number_input(
                    "Heart Rate (bpm)",
                    min_value=40,
                    max_value=220,
                    value=75,
                    key="manual_hr",
                )
                rmssd_manual = st.number_input(
                    "HRV (RMSSD, ms)",
                    min_value=5.0,
                    max_value=200.0,
                    value=42.0,
                    step=0.5,
                    key="manual_rmssd",
                )
                steps_manual = st.number_input(
                    "Steps (today)",
                    min_value=0,
                    max_value=100000,
                    value=4000,
                    step=100,
                    key="manual_steps",
                )
                sleep_manual = st.number_input(
                    "Sleep Hours (last night)",
                    min_value=0.0,
                    max_value=16.0,
                    value=7.0,
                    step=0.25,
                    key="manual_sleep_hours",
                )

                if st.button("Submit", key="manual_submit", use_container_width=True):
                    st.session_state.manual_entry = {
                        "hr": hr_manual,
                        "rmssd": rmssd_manual,
                        "steps": steps_manual,
                        "sleep_hours": sleep_manual,
                    }
                    st.success(
                        "Manual values captured. For detailed analysis, you can also use the **Predict Stress** page."
                    )

    with tab_realtime:
        st.markdown(
            '<div class="section-header">📡 Real-Time Continuous Monitoring</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#8892a4;">Stream heart rate and HRV in real time from Fitbit or simulate a live feed from a CSV export. '
            'Data is processed in sliding 60-second windows (step 10 seconds), features are extracted, a trained ML model predicts stress level, '
            'and results are logged to a local SQLite database for trend analysis.</p>',
            unsafe_allow_html=True,
        )

        with st.expander("How real-time monitoring works", expanded=False):
            st.markdown(
                "**Fitbit API:**\n"
                "1. **Wear your Fitbit** so it can record heart rate.\n"
                "2. **Sync your device** in the Fitbit app (or let it auto-sync). Fitbit only exposes intraday HR after sync.\n"
                "3. **Click Start** — the app will fetch today’s intraday heart rate and HRV every few seconds, fill a 60s window, then run the stress model and show the result.\n\n"
                "**CSV Upload:** Choose a CSV with timestamp, HR, and (optionally) RMSSD/SCL. Click Start to simulate a live stream from the file."
            )

        # Local styling for real-time cards and status indicators
        st.markdown(
            """
            <style>
            .rt-card {
                border-radius: 16px;
                padding: 1rem 1.2rem;
                background: radial-gradient(circle at top left, rgba(56,189,248,0.15), rgba(15,23,42,0.95));
                border: 1px solid rgba(148,163,184,0.5);
                box-shadow: 0 18px 40px rgba(15,23,42,0.8);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
            }
            .rt-card-title {
                font-size: 0.85rem;
                font-weight: 600;
                color: #e5e7eb;
                margin-bottom: 0.4rem;
            }
            .rt-meta-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.75rem;
                font-size: 0.75rem;
                color: #9ca3af;
            }
            .rt-meta-pill {
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                background: rgba(15,23,42,0.7);
                border: 1px solid rgba(148,163,184,0.6);
            }
            .rt-status-dot {
                width: 8px;
                height: 8px;
                border-radius: 999px;
                display: inline-block;
                margin-right: 0.35rem;
                background: #6b7280;
            }
            .rt-status-dot-active {
                background: #22c55e;
                box-shadow: 0 0 0 0 rgba(34,197,94,0.8);
                animation: rt-pulse 1.4s infinite;
            }
            @keyframes rt-pulse {
                0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.8); }
                70% { box-shadow: 0 0 0 10px rgba(34,197,94,0); }
                100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
            }
            .rt-chart-card {
                border-radius: 16px;
                padding: 0.8rem 0.9rem;
                background: rgba(15,23,42,0.9);
                border: 1px solid rgba(31,41,55,0.9);
                box-shadow: 0 12px 30px rgba(15,23,42,0.85);
            }
            .rt-chart-title {
                font-size: 0.8rem;
                color: #9ca3af;
                margin-bottom: 0.35rem;
            }
            .rt-chart-placeholder {
                border-radius: 12px;
                border: 1px dashed rgba(55,65,81,0.9);
                height: 150px;
                background: repeating-linear-gradient(
                    135deg,
                    rgba(31,41,55,0.8),
                    rgba(31,41,55,0.8) 6px,
                    rgba(17,24,39,0.95) 6px,
                    rgba(17,24,39,0.95) 12px
                );
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if not HAS_AUTOREFRESH:
            st.warning(
                "Install `streamlit-autorefresh` to enable automatic updates every 5 seconds:\n\n"
                "`pip install streamlit-autorefresh`"
            )

        source = st.radio(
            "Data source",
            ["Fitbit API", "CSV Upload (simulated stream)"],
            index=0 if st.session_state.rt_source == "Fitbit API" else 1,
            horizontal=True,
        )
        st.session_state.rt_source = source

        # ── Device status card + controls ──
        has_connection = (
            (source == "Fitbit API" and bool(st.session_state.get("fitbit_token")))
            or (source == "CSV Upload (simulated stream)" and st.session_state.get("rt_csv_df") is not None)
        )
        has_buffer_data = bool(st.session_state.get("rt_buffer"))
        streaming_label = "Streaming" if st.session_state.rt_monitoring and has_buffer_data else "Waiting for Data"

        status_col, button_col = st.columns([2, 1])
        with status_col:
            status_dot_class = "rt-status-dot rt-status-dot-active" if st.session_state.rt_monitoring else "rt-status-dot"
            connection_text = "Connected" if has_connection else "Not Connected"
            st.markdown(
                f"""
                <div class="rt-card">
                    <div class="rt-card-title">
                        <span class="{status_dot_class}"></span>
                        Device Status
                    </div>
                    <div class="rt-meta-row">
                        <span class="rt-meta-pill">Link: {connection_text}</span>
                        <span class="rt-meta-pill">Stream: {streaming_label}</span>
                        <span class="rt-meta-pill">Window Size: 60 s</span>
                        <span class="rt-meta-pill">Step Size: 10 s</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with button_col:
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                start_clicked = st.button(
                    "▶️ Start",
                    type="primary",
                    use_container_width=True,
                    key="rt_start_btn",
                )
            with b_col2:
                stop_clicked = st.button(
                    "⏹️ Stop",
                    use_container_width=True,
                    key="rt_stop_btn",
                )

            if start_clicked and not st.session_state.rt_monitoring:
                st.session_state.rt_monitoring = True
                st.session_state.rt_last_window_end = None
                st.session_state.rt_buffer = []
                if source == "CSV Upload (simulated stream)" and st.session_state.rt_csv_df is not None:
                    # Reset CSV playback cursor to the beginning of the file
                    ts_col = st.session_state.rt_csv_ts_col
                    df_csv = st.session_state.rt_csv_df
                    df_csv["timestamp"] = pd.to_datetime(df_csv[ts_col], errors="coerce")
                    df_csv = df_csv.dropna(subset=["timestamp"]).sort_values("timestamp")
                    st.session_state.rt_csv_df = df_csv
                    if not df_csv.empty:
                        st.session_state.rt_csv_cursor = df_csv["timestamp"].min()
                st.rerun()

            if stop_clicked and st.session_state.rt_monitoring:
                st.session_state.rt_monitoring = False
                st.rerun()

        st.markdown("")

        # CSV configuration (used when source == CSV)
        if source == "CSV Upload (simulated stream)":
            uploaded_rt_file = st.file_uploader(
                "Upload high-frequency smartwatch CSV (with timestamp, HR, RMSSD, Respiration, Temp, optional movement/ACC)",
                type=["csv"],
                key="rt_csv_uploader",
            )
            if uploaded_rt_file is not None:
                try:
                    df_rt = pd.read_csv(uploaded_rt_file)
                    st.dataframe(df_rt.head(10), use_container_width=True, height=200)
                    all_cols = df_rt.columns.tolist()
                    ts_candidates = [c for c in all_cols if "time" in c.lower() or "timestamp" in c.lower()]
                    hr_candidates = [c for c in all_cols if any(x in c.lower() for x in ["hr", "heart", "bpm", "rate"])]
                    rmssd_candidates = [c for c in all_cols if any(x in c.lower() for x in ["rmssd", "hrv"])]
                    resp_candidates = [c for c in all_cols if any(x in c.lower() for x in ["resp", "breath"])]
                    temp_candidates = [c for c in all_cols if any(x in c.lower() for x in ["temp", "skin"])]
                    move_candidates = [c for c in all_cols if any(x in c.lower() for x in ["acc", "accelerometer", "movement", "activity"])]

                    ts_col = st.selectbox(
                        "Timestamp column",
                        options=all_cols,
                        index=all_cols.index(ts_candidates[0]) if ts_candidates else 0,
                    )
                    hr_col = st.selectbox(
                        "Heart rate column",
                        options=all_cols,
                        index=all_cols.index(hr_candidates[0]) if hr_candidates else 0,
                    )
                    rmssd_col = st.selectbox(
                        "RMSSD (HRV) column",
                        options=all_cols,
                        index=all_cols.index(rmssd_candidates[0]) if rmssd_candidates else 0,
                    )
                    resp_col = st.selectbox(
                        "Respiration column (optional)",
                        options=["(none)"] + all_cols,
                        index=0 if not resp_candidates else all_cols.index(resp_candidates[0]) + 1,
                    )
                    temp_col = st.selectbox(
                        "Temperature column (optional)",
                        options=["(none)"] + all_cols,
                        index=0 if not temp_candidates else all_cols.index(temp_candidates[0]) + 1,
                    )
                    move_col = st.selectbox(
                        "Movement / ACC column (optional)",
                        options=["(none)"] + all_cols,
                        index=0 if not move_candidates else all_cols.index(move_candidates[0]) + 1,
                    )

                    # Persist mapping for subsequent refreshes
                    st.session_state.rt_csv_df = df_rt
                    st.session_state.rt_csv_ts_col = ts_col
                    st.session_state.rt_csv_hr_col = hr_col
                    st.session_state.rt_csv_rmssd_col = rmssd_col
                    st.session_state.rt_csv_resp_col = None if resp_col == "(none)" else resp_col
                    st.session_state.rt_csv_temp_col = None if temp_col == "(none)" else temp_col
                    st.session_state.rt_csv_move_col = None if move_col == "(none)" else move_col
                except Exception as e:
                    st.error(f"Error reading real-time CSV: {e}")

        # Auto-refresh every 5 seconds while monitoring
        if st.session_state.rt_monitoring and HAS_AUTOREFRESH:
            st_autorefresh(interval=5000, key="rt_monitor_refresh")

        # ── Real-time ingestion helpers (inline for clarity on this page) ──
        def _update_buffer_from_fitbit_live():
            """Append the latest Fitbit HR/RMSSD sample into the in-memory real-time buffer."""
            try:
                import requests  # Local import to avoid hard dependency if not used
            except ImportError:
                st.warning("Install `requests` for Fitbit integration: `pip install requests`")
                return

            fitbit_client_id = _get_secret_or_env("FITBIT_CLIENT_ID")
            fitbit_client_secret = _get_secret_or_env("FITBIT_CLIENT_SECRET")
            if not fitbit_client_id or not fitbit_client_secret:
                st.info("Fitbit API keys not configured. Configure them in `.streamlit/secrets.toml` to use live monitoring.")
                return

            if "fitbit_token" not in st.session_state or not st.session_state.fitbit_token:
                st.info("Connect Fitbit on the previous tab first, then return here to enable live monitoring.")
                return

            access_token = _ensure_fitbit_access_token()
            if not access_token:
                st.info("Fitbit session is not available. Please reconnect your device from the Smartwatch page.")
                return

            headers = {"Authorization": "Bearer " + access_token}
            today = datetime.now().strftime("%Y-%m-%d")
            try:
                # Use intraday heart rate if available for higher temporal resolution
                hr_resp = requests.get(
                    f"https://api.fitbit.com/1/user/-/activities/heart/date/{today}/1d/1sec.json",
                    headers=headers,
                    timeout=10,
                )
                hrv_resp = requests.get(
                    f"https://api.fitbit.com/1/user/-/hrv/date/{today}.json",
                    headers=headers,
                    timeout=10,
                )
            except Exception as e:
                st.error(f"Fitbit API error: {e}")
                return

            if hr_resp.status_code != 200:
                st.warning("Unable to fetch live heart rate from Fitbit.")
                return

            hr_json = hr_resp.json()
            ds = (
                hr_json.get("activities-heart-intraday", {})
                .get("dataset", [])
            )
            if not ds:
                st.info(
                    "**No intraday heart rate data yet.** Fitbit only provides live-style data after you sync. "
                    "Wear your device, open the Fitbit app and sync, then click **Start** so we can fetch today’s data."
                )
                return

            last_point = ds[-1]
            hr_val = last_point.get("value")
            ts_str = last_point.get("time")
            # Combine Fitbit-provided clock time with today's date
            ts = datetime.strptime(f"{today} {ts_str}", "%Y-%m-%d %H:%M:%S")

            rmssd_val = None
            if hrv_resp.status_code == 200:
                hrv_json = hrv_resp.json().get("hrv", [{}])
                if hrv_json and hrv_json[0].get("value", {}).get("dailyRmssd"):
                    rmssd_val = hrv_json[0]["value"]["dailyRmssd"]

            if hr_val is None and rmssd_val is None:
                return

            record = {
                "timestamp": ts,
                "hr": float(hr_val) if hr_val is not None else None,
                "rmssd": float(rmssd_val) if rmssd_val is not None else None,
                "respiration": None,
                "temp": None,
                "movement": None,
            }
            st.session_state.rt_buffer.append(record)

        def _update_buffer_from_csv_stream():
            """Advance a virtual cursor through the uploaded CSV and fill the 60-second sliding window."""
            df_csv = st.session_state.rt_csv_df
            if df_csv is None:
                return

            # Ensure timestamp column is always proper datetime for windowing
            # Some CSVs may re-load with string timestamps, which would break comparisons.
            if "timestamp" in df_csv.columns:
                ts_series = df_csv["timestamp"]
            else:
                ts_raw_col = st.session_state.get("rt_csv_ts_col")
                if not ts_raw_col or ts_raw_col not in df_csv.columns:
                    return
                ts_series = df_csv[ts_raw_col]

            if not np.issubdtype(ts_series.dtype, np.datetime64):
                df_csv = df_csv.copy()
                df_csv["timestamp"] = pd.to_datetime(ts_series, errors="coerce")
                df_csv = df_csv.dropna(subset=["timestamp"]).sort_values("timestamp")
                st.session_state.rt_csv_df = df_csv

            cursor = st.session_state.rt_csv_cursor
            if cursor is None:
                # Will be initialized when monitoring starts
                return

            # Define current 60-second window based on cursor
            window_start = cursor
            window_end = cursor + timedelta(seconds=60)
            ts_col = "timestamp"
            mask = (df_csv[ts_col] >= window_start) & (df_csv[ts_col] < window_end)
            window_df = df_csv.loc[mask]
            if window_df.empty:
                # No more data to stream
                st.session_state.rt_monitoring = False
                return

            # Map columns into buffer records
            hr_col = st.session_state.rt_csv_hr_col
            rmssd_col = st.session_state.rt_csv_rmssd_col
            resp_col = st.session_state.rt_csv_resp_col
            temp_col = st.session_state.rt_csv_temp_col
            move_col = st.session_state.rt_csv_move_col

            for _, row in window_df.iterrows():
                record = {
                    "timestamp": row[ts_col],
                    "hr": float(row[hr_col]) if hr_col in row and pd.notna(row[hr_col]) else None,
                    "rmssd": float(row[rmssd_col]) if rmssd_col in row and pd.notna(row[rmssd_col]) else None,
                    "respiration": float(row[resp_col]) if resp_col and resp_col in row and pd.notna(row[resp_col]) else None,
                    "temp": float(row[temp_col]) if temp_col and temp_col in row and pd.notna(row[temp_col]) else None,
                    "movement": float(row[move_col]) if move_col and move_col in row and pd.notna(row[move_col]) else None,
                }
                st.session_state.rt_buffer.append(record)

            # Advance virtual cursor by step size (10 seconds)
            st.session_state.rt_csv_cursor = cursor + timedelta(seconds=10)

        def _compute_window_features(records):
            """Compute window-level features from the latest 60 seconds of samples."""
            if not records:
                return None
            df = pd.DataFrame(records)
            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df.empty:
                return None

            latest_ts = df["timestamp"].max()
            window_start = latest_ts - timedelta(seconds=60)
            df_win = df[df["timestamp"] >= window_start]
            if df_win.empty:
                return None

            # Normalize HR and RMSSD inside the window (simple min-max scaling)
            def _norm(series):
                series = pd.to_numeric(series, errors="coerce").dropna()
                if series.empty:
                    return series
                mn, mx = series.min(), series.max()
                if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
                    return series - mn
                return (series - mn) / (mx - mn)

            hr_series = pd.to_numeric(df_win.get("hr"), errors="coerce")
            rmssd_series = pd.to_numeric(df_win.get("rmssd"), errors="coerce")
            hr_norm = _norm(hr_series)
            rmssd_norm = _norm(rmssd_series)

            mean_hr = float(hr_norm.mean()) if not hr_norm.empty else np.nan
            rmssd_feature = float(rmssd_norm.mean()) if not rmssd_norm.empty else np.nan
            hrv_var = float(rmssd_norm.var()) if not rmssd_norm.empty else np.nan

            respiration_series = pd.to_numeric(df_win.get("respiration"), errors="coerce")
            respiration_rate = float(respiration_series.mean()) if respiration_series.notna().any() else np.nan

            movement_series = pd.to_numeric(df_win.get("movement"), errors="coerce")
            movement_intensity = float(movement_series.mean()) if movement_series.notna().any() else np.nan

            return {
                "timestamp": latest_ts,
                "mean_hr": mean_hr,
                "rmssd": rmssd_feature,
                "hrv_var": hrv_var,
                "respiration": respiration_rate,
                "movement_intensity": movement_intensity,
            }

        def _predict_stress_level(features):
            """Load trained ML model and predict Low / Medium / High stress."""
            if features is None:
                return None
            model_path_rt = os.path.join(MODEL_DIR, "stress_model.pkl")
            if not os.path.exists(model_path_rt):
                st.warning("Real-time model `stress_model.pkl` not found in the `models` folder.")
                return None

            try:
                model_rt = joblib.load(model_path_rt)
            except Exception as e:
                st.error(f"Unable to load real-time model: {e}")
                return None

            x = np.array(
                [
                    [
                        features["mean_hr"],
                        features["rmssd"],
                        features["hrv_var"],
                        features["respiration"],
                        features["movement_intensity"],
                    ]
                ]
            )
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            label = None
            try:
                if hasattr(model_rt, "predict_proba"):
                    probs = model_rt.predict_proba(x)[0]
                    idx = int(np.argmax(probs))
                    if idx == 0:
                        label = "Low"
                    elif idx == 1:
                        label = "Medium"
                    else:
                        label = "High"
                else:
                    raw = model_rt.predict(x)[0]
                    if isinstance(raw, (int, float)):
                        if raw <= 0:
                            label = "Low"
                        elif raw == 1:
                            label = "Medium"
                        else:
                            label = "High"
                    else:
                        s = str(raw).lower()
                        if "low" in s:
                            label = "Low"
                        elif "med" in s:
                            label = "Medium"
                        else:
                            label = "High"
            except Exception as e:
                st.error(f"Error during real-time prediction: {e}")
                return None

            return label

        # ── Run one monitoring step per rerun ──
        current_features = None
        current_label = None
        if st.session_state.rt_monitoring:
            # 1) Ingest new data into buffer
            if source == "Fitbit API":
                _update_buffer_from_fitbit_live()
            else:
                _update_buffer_from_csv_stream()

            # Trim buffer to last 60 seconds to bound memory
            if st.session_state.rt_buffer:
                buf_df = pd.DataFrame(st.session_state.rt_buffer)
                buf_df["timestamp"] = pd.to_datetime(buf_df["timestamp"], errors="coerce")
                buf_df = buf_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                latest_ts = buf_df["timestamp"].max()
                cutoff_ts = latest_ts - timedelta(seconds=60)
                buf_df = buf_df[buf_df["timestamp"] >= cutoff_ts]
                st.session_state.rt_buffer = buf_df.to_dict("records")

            # 2) Every 10 seconds, compute a window and predict
            now_ts = datetime.utcnow()
            if (
                st.session_state.rt_last_window_end is None
                or (now_ts - st.session_state.rt_last_window_end).total_seconds() >= 10
            ):
                feats = _compute_window_features(st.session_state.rt_buffer)
                current_features = feats
                if feats is not None:
                    label = _predict_stress_level(feats)
                    if label is not None:
                        current_label = label
                        st.session_state.rt_last_window_end = now_ts
                        # Persist to SQLite
                        log_stress_prediction_to_db(
                            feats["timestamp"],
                            feats["mean_hr"],
                            feats["rmssd"],
                            feats["respiration"],
                            label,
                        )

        # ── Current status + dashboard ──
        st.markdown("### Current Stress")
        cur_col1, cur_col2 = st.columns([1, 2])
        with cur_col1:
            if current_label is None:
                st.markdown(
                    """
                    <div class="rt-card" style="padding:1rem 1.1rem;">
                        <div style="font-size:0.8rem;color:#9ca3af;">Current Stress Level</div>
                        <div style="font-size:0.9rem;color:#6b7280;margin-top:0.35rem;">
                            No prediction yet. Waiting for enough data in the 60s window.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                color = "#3fb950" if current_label == "Low" else "#d29922" if current_label == "Medium" else "#f85149"
                st.markdown(
                    f"""
                    <div class="rt-card" style="padding:1.2rem 1.4rem;">
                        <div style="font-size:0.8rem;color:#9ca3af;">Current Stress Level</div>
                        <div style="font-size:2rem;font-weight:700;color:{color};margin-top:0.3rem;">
                            {current_label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with cur_col2:
            last_hour_df = load_last_hour_stress_history()
            if last_hour_df is not None and not last_hour_df.empty:
                last_hour_df = last_hour_df.copy()
                last_hour_df["timestamp"] = pd.to_datetime(last_hour_df["timestamp"], errors="coerce")
                last_hour_df = last_hour_df.dropna(subset=["timestamp"])
                # Map stress level to numeric for plotting
                level_map = {"Low": 0, "Medium": 1, "High": 2}
                last_hour_df["stress_score"] = last_hour_df["stress_level"].map(level_map)
                fig_trend = px.line(
                    last_hour_df,
                    x="timestamp",
                    y="stress_score",
                    markers=True,
                    title="Last 1 Hour Stress Trend",
                )
                fig_trend.update_yaxes(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=["Low", "Medium", "High"],
                    range=[-0.1, 2.1],
                )
                fig_trend.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=260,
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.markdown(
                    """
                    <div class="rt-card" style="padding:0.9rem 1rem;">
                        <div style="font-size:0.8rem;color:#9ca3af;margin-bottom:0.25rem;">Last 1 Hour Stress Trend</div>
                        <div style="font-size:0.8rem;color:#6b7280;">
                            No stress history yet. Start monitoring to populate the trend chart.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── Live signal placeholders (HR, RMSSD, Stress Score) ──
        st.markdown("")
        st.markdown("### Live Signals")
        sig_col1, sig_col2, sig_col3 = st.columns(3)

        def _render_empty_signal(title):
            st.markdown(
                f"""
                <div class="rt-chart-card">
                    <div class="rt-chart-title">{title}</div>
                    <div class="rt-chart-placeholder"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        buf_df_live = None
        if st.session_state.get("rt_buffer"):
            try:
                buf_df_live = pd.DataFrame(st.session_state.rt_buffer)
                if "timestamp" in buf_df_live:
                    buf_df_live["timestamp"] = pd.to_datetime(buf_df_live["timestamp"], errors="coerce")
                    buf_df_live = buf_df_live.dropna(subset=["timestamp"]).sort_values("timestamp")
            except Exception:
                buf_df_live = None

        # Heart rate
        with sig_col1:
            if buf_df_live is None or "hr" not in buf_df_live or buf_df_live["hr"].dropna().empty:
                _render_empty_signal("Live Heart Rate")
            else:
                fig_hr = px.line(
                    buf_df_live,
                    x="timestamp",
                    y="hr",
                    title="Live Heart Rate",
                )
                fig_hr.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=190)
                st.plotly_chart(fig_hr, use_container_width=True)

        # HRV (RMSSD)
        with sig_col2:
            if buf_df_live is None or "rmssd" not in buf_df_live or buf_df_live["rmssd"].dropna().empty:
                _render_empty_signal("Live HRV (RMSSD)")
            else:
                fig_hrv = px.line(
                    buf_df_live,
                    x="timestamp",
                    y="rmssd",
                    title="Live HRV (RMSSD)",
                )
                fig_hrv.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=190)
                st.plotly_chart(fig_hrv, use_container_width=True)

        # Live Stress Score (numeric from last-hour history)
        with sig_col3:
            last_hour_df_live = load_last_hour_stress_history()
            if (
                last_hour_df_live is None
                or last_hour_df_live.empty
                or "stress_level" not in last_hour_df_live
            ):
                _render_empty_signal("Live Stress Score")
            else:
                last_hour_df_live = last_hour_df_live.copy()
                last_hour_df_live["timestamp"] = pd.to_datetime(last_hour_df_live["timestamp"], errors="coerce")
                last_hour_df_live = last_hour_df_live.dropna(subset=["timestamp"]).sort_values("timestamp")
                level_map_live = {"Low": 10, "Medium": 45, "High": 80}
                last_hour_df_live["stress_score_live"] = last_hour_df_live["stress_level"].map(level_map_live)
                fig_live = px.line(
                    last_hour_df_live,
                    x="timestamp",
                    y="stress_score_live",
                    title="Live Stress Score",
                )
                fig_live.update_yaxes(range=[0, 100])
                fig_live.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=190)
                st.plotly_chart(fig_live, use_container_width=True)

        # ── High-stress alert (last 5 minutes) ──
        hist_df = load_last_hour_stress_history()
        if hist_df is not None and not hist_df.empty:
            hist_df = hist_df.copy()
            hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"], errors="coerce")
            hist_df = hist_df.dropna(subset=["timestamp"])
            if not hist_df.empty:
                latest_ts = hist_df["timestamp"].max()
                last5 = hist_df[hist_df["timestamp"] >= latest_ts - timedelta(minutes=5)]
                if not last5.empty and (last5["stress_level"] == "High").all():
                    st.error(
                        "🚨 **High stress detected continuously for more than 5 minutes.** "
                        "Consider taking a short break, breathing exercise, or a quick walk."
                    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    render_topbar("Model Results")
    st.markdown('<h1 class="main-title">Model Results</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Visualizations of ML and DL model performance across all approaches</p>',
        unsafe_allow_html=True,
    )

    # Approach selector
    approach = st.selectbox(
        "Select Approach",
        [
            "Generic Models",
            "Generic Models (with ID)",
            "Single Task Learning (DL)",
            "User-Based Splitting",
            "MTL User-as-Task",
            "Single-Attribute Splitting",
            "MTL Single-Attribute-Group",
            "Multi-Attribute Splitting",
            "MTL Multi-Attribute-Group",
            "Fuzzy Clustering Splitting",
            "MTL Fuzzy-Group",
            "Clustering Analysis",
        ],
    )

    # Map approach to directory
    approach_dirs = {
        "Generic Models": "Results/Generic/Generic",
        "Generic Models (with ID)": "Results/Generic/GenericID",
        "Single Task Learning (DL)": "Results/Generic/STL",
        "User-Based Splitting": "Results/User-based-models/User-based-splitting",
        "MTL User-as-Task": "Results/User-based-models/MTL-user-as-task",
        "Single-Attribute Splitting": "Results/Single-attribute-based-models/Single-attribute-splitting",
        "MTL Single-Attribute-Group": "Results/Single-attribute-based-models/MTL-Single-attribute-group",
        "Multi-Attribute Splitting": "Results/Multi-attribute-based-models/Multi-attribute-splitting",
        "MTL Multi-Attribute-Group": "Results/Multi-attribute-based-models/MTL-Multi-attribute-group",
        "Fuzzy Clustering Splitting": "Results/Fuzzy-based-models/Fuzzy-clustering-splitting",
        "MTL Fuzzy-Group": "Results/Fuzzy-based-models/MTL-fuzzy-group",
        "Clustering Analysis": "Results/Clustering",
    }

    approach_descriptions = {
        "Generic Models": "Person-independent models trained on all data. Uses LDA, SVM, Logistic Regression, Random Forest, etc.",
        "Generic Models (with ID)": "Same as Generic, but with participant ID as an additional feature.",
        "Single Task Learning (DL)": "Generic binary classification neural network (single-task learning) for each dataset.",
        "User-Based Splitting": "Personalized ML models built separately for each participant.",
        "MTL User-as-Task": "Multitask learning — each user's stress detection is treated as a separate task.",
        "Single-Attribute Splitting": "Users grouped by personality via K-means clustering. Separate models per group.",
        "MTL Single-Attribute-Group": "MTL with personality-based group as a task.",
        "Multi-Attribute Splitting": "Users grouped by multiple attributes (caffeine, exercise, sleep, age, gender, physiology stats).",
        "MTL Multi-Attribute-Group": "MTL with multi-attribute group as a task — achieves the best overall results.",
        "Fuzzy Clustering Splitting": "Users assigned to clusters with membership degrees via Fuzzy C-Means.",
        "MTL Fuzzy-Group": "MTL using fuzzy cluster membership for task assignment.",
        "Clustering Analysis": "Elbow method, silhouette scores, and t-SNE visualizations for cluster selection.",
    }

    st.info(approach_descriptions.get(approach, ""))

    result_dir = os.path.join(BASE_DIR, approach_dirs[approach])
    if os.path.exists(result_dir):
        # Get all PNG files
        png_files = sorted(glob.glob(os.path.join(result_dir, "*.png")))
        csv_files = sorted(glob.glob(os.path.join(result_dir, "*.csv")))

        if png_files:
            # Try to display in a grid of 2 columns
            cols = st.columns(2)
            for i, img_path in enumerate(png_files):
                with cols[i % 2]:
                    fname = os.path.basename(img_path)
                    # Create a nice label
                    label = fname.replace("_", " ").replace(".png", "").title()
                    st.image(img_path, caption=label, use_container_width=True)
        else:
            st.info("No PNG results found for this approach.")

        # Show CSV data if available
        if csv_files:
            st.markdown('<div class="section-header">📄 Training Metrics (CSV)</div>', unsafe_allow_html=True)
            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                with st.expander(f"📄 {fname}"):
                    try:
                        df = pd.read_csv(csv_path)
                        st.dataframe(df, use_container_width=True, height=300)
                    except Exception as e:
                        st.error(f"Error loading {fname}: {e}")
    else:
        st.warning(f"Directory not found: {result_dir}")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Data Explorer":
    render_topbar("Data Explorer")
    st.markdown('<h1 class="main-title">Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Interactive exploration of the physiological datasets</p>',
        unsafe_allow_html=True,
    )

    # Dataset selector
    csv_files = {
        "SWELL-KW": "swell_new.csv",  # may be overridden by swell_clean.csv
        "LifeSnaps": "lifesnaps_new.csv",
        "LifeSnaps (with time)": "lifesnaps_time.csv",
        "Daily Fitbit (raw)": "daily_fitbit_sema_df_unprocessed.csv",
    }

    selected_dataset = st.selectbox("Select Dataset", list(csv_files.keys()))

    @st.cache_data
    def load_csv(path):
        return pd.read_csv(path)

    # Prefer cleaned SWELL-KW file if present
    if selected_dataset == "SWELL-KW":
        clean_path = os.path.join(CSV_DIR, "swell_clean.csv")
        if os.path.exists(clean_path):
            csv_path = clean_path
            st.caption("Using cleaned SWELL-KW dataset (swell_clean.csv).")
        else:
            csv_path = os.path.join(CSV_DIR, csv_files[selected_dataset])
    else:
        csv_path = os.path.join(CSV_DIR, csv_files[selected_dataset])

    try:
        df = load_csv(csv_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Overview metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Rows", f"{len(df):,}")
    with m2:
        st.metric("Columns", len(df.columns))
    with m3:
        if "id" in df.columns:
            st.metric("Participants", df["id"].nunique())
        else:
            st.metric("Participants", "N/A")
    with m4:
        if "stress" in df.columns:
            stress_ratio = df["stress"].mean() * 100
            st.metric("Stress %", f"{stress_ratio:.1f}%")
        else:
            st.metric("Stress %", "N/A")

    # Tabs
    tab_data, tab_dist, tab_corr, tab_stress = st.tabs(
        ["📋 Data Preview", "📊 Distributions", "🔗 Correlations", "😰 Stress Analysis"]
    )

    with tab_data:
        st.dataframe(df.head(100), use_container_width=True, height=400)

        st.markdown("**Column Statistics**")
        st.dataframe(df.describe().round(3), use_container_width=True)

    with tab_dist:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove id, dataset columns from numeric for visualization
        plot_cols = [c for c in numeric_cols if c not in ["id", "dataset", "stress"]]

        if plot_cols:
            selected_col = st.selectbox("Select Feature", plot_cols)

            fig_hist = px.histogram(
                df, x=selected_col,
                color="stress" if "stress" in df.columns else None,
                marginal="box",
                title=f"Distribution of {selected_col}",
                color_discrete_sequence=["#00d4aa", "#ff4b4b"],
                template="plotly_dark",
                opacity=0.7,
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Box plot by stress
            if "stress" in df.columns:
                fig_box = px.box(
                    df, x="stress", y=selected_col,
                    color="stress",
                    title=f"{selected_col} by Stress Label",
                    color_discrete_sequence=["#00d4aa", "#ff4b4b"],
                    template="plotly_dark",
                )
                fig_box.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                )
                st.plotly_chart(fig_box, use_container_width=True)

    with tab_corr:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        plot_cols = [c for c in numeric_cols if c not in ["id", "dataset"]]

        if len(plot_cols) > 1:
            # Limit columns for readability
            if len(plot_cols) > 20:
                st.info(f"Showing top 20 features (out of {len(plot_cols)}) by variance.")
                variances = df[plot_cols].var().sort_values(ascending=False)
                plot_cols = variances.head(20).index.tolist()

            corr = df[plot_cols].corr()
            fig_corr = px.imshow(
                corr,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Feature Correlation Matrix",
                template="plotly_dark",
            )
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab_stress:
        if "stress" in df.columns:
            # Stress distribution
            stress_counts = df["stress"].value_counts().reset_index()
            stress_counts.columns = ["Stress", "Count"]
            stress_counts["Label"] = stress_counts["Stress"].map({0: "No Stress", 1: "Stress"})

            fig_pie = px.pie(
                stress_counts, values="Count", names="Label",
                color_discrete_sequence=["#00d4aa", "#ff4b4b"],
                title="Stress Label Distribution",
                template="plotly_dark",
                hole=0.4,
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Per-participant stress ratio
            if "id" in df.columns:
                per_user = df.groupby("id")["stress"].mean().reset_index()
                per_user.columns = ["Participant", "Stress Ratio"]
                per_user = per_user.sort_values("Stress Ratio", ascending=False)

                fig_user = px.bar(
                    per_user, x="Participant", y="Stress Ratio",
                    title="Stress Ratio per Participant",
                    color="Stress Ratio",
                    color_continuous_scale=["#00d4aa", "#ffc832", "#ff4b4b"],
                    template="plotly_dark",
                )
                fig_user.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                )
                st.plotly_chart(fig_user, use_container_width=True)
        else:
            st.info("No 'stress' column found in this dataset.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LONG-TERM ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Long-Term Analytics":
    render_topbar("Long-Term Analytics")
    st.markdown('<h1 class="main-title">Long-Term Health Analytics</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Track stress patterns over time · Recovery insights · Pre-emptive alerts</p>',
        unsafe_allow_html=True,
    )

    # ── Helper: generate sample data for demo ──
    def generate_sample_data(n=30):
        """Generate realistic sample stress readings spanning a simulated week."""
        samples = []
        base_time = datetime.now() - timedelta(days=6, hours=23)
        for i in range(n):
            # Simulate stress patterns: morning calm → afternoon stress → evening recovery
            time_offset = timedelta(hours=random.uniform(0, 168))  # across 7 days
            ts = base_time + timedelta(hours=i * (168 / n))

            # Create realistic wave patterns
            hour_of_day = ts.hour
            day_of_week = ts.weekday()

            # Higher stress on weekdays during work hours
            base_stress = 30
            if day_of_week < 5:  # weekday
                if 9 <= hour_of_day <= 17:
                    base_stress = 55 + random.uniform(-10, 20)
                elif 17 < hour_of_day <= 21:
                    base_stress = 35 + random.uniform(-5, 10)
            else:  # weekend
                base_stress = 25 + random.uniform(-5, 10)

            # Add a stress spike on day 4-5 (simulating burnout)
            days_elapsed = (ts - base_time).total_seconds() / 86400
            if 3.5 <= days_elapsed <= 5.0:
                base_stress += 15

            stress_prob = max(5, min(95, base_stress + random.uniform(-8, 8)))
            hr = int(65 + (stress_prob / 100) * 50 + random.uniform(-5, 5))
            rmssd = max(10, 80 - (stress_prob / 100) * 60 + random.uniform(-5, 5))
            scl = max(0.5, 2 + (stress_prob / 100) * 18 + random.uniform(-1, 1))

            samples.append({
                "timestamp": ts,
                "hr": hr,
                "rmssd": round(rmssd, 1),
                "scl": round(scl, 1),
                "stress_prob": round(stress_prob, 1),
                "prediction": 1 if stress_prob > 50 else 0,
            })
        return samples

    # ── Controls row ──
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
    with ctrl_col1:
        if st.button("🎲 Generate Sample Data (Demo)", use_container_width=True, type="primary"):
            st.session_state.stress_log = generate_sample_data(35)
            save_stress_log_to_csv()
            st.rerun()
    with ctrl_col2:
        st.markdown(
            f'<div class="metric-card" style="padding:0.8rem;">'
            f'<div class="metric-value" style="font-size:1.6rem;">{len(st.session_state.stress_log)}</div>'
            f'<div class="metric-label">Logged Entries</div></div>',
            unsafe_allow_html=True,
        )
    with ctrl_col3:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.stress_log = []
            save_stress_log_to_csv()
            st.rerun()

    st.markdown("")

    # ── Quick Log Form ──
    with st.expander("➕ Quick Log — Add a Reading Manually", expanded=False):
        log_c1, log_c2, log_c3 = st.columns(3)
        with log_c1:
            log_hr = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75, key="log_hr")
        with log_c2:
            log_rmssd = st.number_input("RMSSD (ms)", min_value=5.0, max_value=200.0, value=42.0, step=0.5, key="log_rmssd")
        with log_c3:
            log_scl = st.number_input("SCL (µS)", min_value=0.0, max_value=40.0, value=5.0, step=0.1, key="log_scl")

        if st.button("📝 Log This Reading", use_container_width=True):
            # Estimate stress probability from features (simple heuristic)
            hr_score = max(0, min(100, (log_hr - 60) / 80 * 100))
            rmssd_score = max(0, min(100, (80 - log_rmssd) / 70 * 100))
            scl_score = max(0, min(100, (log_scl - 2) / 18 * 100))
            estimated_stress = (hr_score * 0.35 + rmssd_score * 0.35 + scl_score * 0.30)
            estimated_stress = max(5, min(95, estimated_stress))

            st.session_state.stress_log.append({
                "timestamp": datetime.now(),
                "hr": log_hr,
                "rmssd": log_rmssd,
                "scl": log_scl,
                "stress_prob": round(estimated_stress, 1),
                "prediction": 1 if estimated_stress > 50 else 0,
            })
            save_stress_log_to_csv()
            st.rerun()

    # ── Main content ──
    if len(st.session_state.stress_log) < 3:
        st.markdown(
            '<div class="prediction-box" style="background:linear-gradient(135deg,#1a1f2e,#252d3d);'
            'border:2px dashed rgba(123,104,238,0.3);">'
            '<div style="font-size:3rem;">📊</div>'
            '<div style="font-size:1.3rem; font-weight:600; color:#8892a4; margin-top:0.5rem;">'
            'Log at least 3 readings to see analytics<br>'
            '<span style="font-size:0.9rem;">Use "Generate Sample Data" for a quick demo, or '
            'make predictions on the Predict Stress page</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        log_df = pd.DataFrame(st.session_state.stress_log)
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
        log_df = log_df.sort_values("timestamp").reset_index(drop=True)
        log_df["entry_num"] = range(1, len(log_df) + 1)
        # Derive calendar date for daily aggregation (used by sleep analytics)
        log_df["date"] = log_df["timestamp"].dt.date

        # ════════════════════════════════════════════════════════════════
        # SECTION 1: WEEKLY STRESS REPORT & RECOVERY SCORE
        # ════════════════════════════════════════════════════════════════
        st.markdown(
            '<div class="section-header">📅 Weekly Stress Report & Recovery Score</div>',
            unsafe_allow_html=True,
        )

        # ── Summary metrics ──
        avg_stress = log_df["stress_prob"].mean()
        peak_stress = log_df["stress_prob"].max()
        peak_time = log_df.loc[log_df["stress_prob"].idxmax(), "timestamp"]
        min_stress = log_df["stress_prob"].min()
        stress_entries = (log_df["prediction"] == 1).sum()
        calm_entries = (log_df["prediction"] == 0).sum()

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            color = "#ff4b4b" if avg_stress > 50 else "#ffc832" if avg_stress > 35 else "#00d4aa"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color};">{avg_stress:.1f}%</div>'
                f'<div class="metric-label">Avg Stress Level</div></div>',
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#ff4b4b;">{peak_stress:.1f}%</div>'
                f'<div class="metric-label">Peak Stress</div></div>',
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#ff4b4b;">{stress_entries}</div>'
                f'<div class="metric-label">High-Stress Readings</div></div>',
                unsafe_allow_html=True,
            )
        with mc4:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:#00d4aa;">{calm_entries}</div>'
                f'<div class="metric-label">Calm Readings</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Stress Timeline Chart ──
        fig_timeline = go.Figure()

        # Stress-Free Zones (green shading for stress_prob < 40)
        # Moderate Zones (amber shading for 40–70)
        # High-Stress Peaks (red shading for stress_prob ≥ 70)
        fig_timeline.add_trace(go.Scatter(
            x=log_df["timestamp"],
            y=log_df["stress_prob"],
            mode="lines+markers",
            name="Stress Probability",
            line=dict(color="#7b68ee", width=3),
            marker=dict(
                size=8,
                color=log_df["stress_prob"].apply(
                    lambda x: "#ff4b4b" if x >= 60 else "#ffc832" if x >= 20 else "#00d4aa"
                ),
                line=dict(width=1, color="#1a1f2e"),
            ),
            fill="none",
            hovertemplate="<b>%{x}</b><br>Stress: %{y:.1f}%<extra></extra>",
        ))

        # Add threshold lines
        fig_timeline.add_hline(y=60, line_dash="dash", line_color="rgba(255,75,75,0.4)",
                               annotation_text="High Stress Zone", annotation_position="top right",
                               annotation_font_color="#ff4b4b")
        fig_timeline.add_hline(y=20, line_dash="dash", line_color="rgba(0,212,170,0.4)",
                               annotation_text="Stress-Free Zone", annotation_position="bottom right",
                               annotation_font_color="#00d4aa")

        # Add colored background zones
        fig_timeline.add_hrect(y0=60, y1=100, fillcolor="rgba(255,75,75,0.05)", line_width=0)
        fig_timeline.add_hrect(y0=0, y1=40, fillcolor="rgba(0,212,170,0.05)", line_width=0)

        fig_timeline.update_layout(
            title="Stress Level Timeline",
            title_font=dict(color="#fafafa", size=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(
                title="Time", gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#8892a4"), title_font=dict(color="#8892a4"),
            ),
            yaxis=dict(
                title="Stress Probability (%)", range=[0, 100],
                gridcolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#8892a4"), title_font=dict(color="#8892a4"),
            ),
            legend=dict(font=dict(color="#fafafa")),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        # ── Recovery Score Calculation ──
        st.markdown(
            '<div class="analytics-header">💪 Recovery Score</div>',
            unsafe_allow_html=True,
        )

        # Calculate recovery score: how quickly stress drops after high-stress peaks
        stress_probs = log_df["stress_prob"].values
        recovery_scores = []
        for i in range(len(stress_probs) - 1):
            if stress_probs[i] > 55:  # after a stress event
                # How much did it drop in the next reading(s)?
                drop = stress_probs[i] - stress_probs[i + 1]
                if drop > 0:
                    recovery_scores.append(min(100, (drop / stress_probs[i]) * 200))

        if recovery_scores:
            recovery_score = np.mean(recovery_scores)
        else:
            # If no high-stress events, score based on overall calm
            recovery_score = max(0, 100 - avg_stress)

        recovery_score = round(min(100, max(0, recovery_score)), 0)

        rec_col1, rec_col2 = st.columns([1, 2])
        with rec_col1:
            # Recovery Score gauge
            if recovery_score >= 70:
                rec_color = "#00d4aa"
                rec_label = "Excellent Recovery"
                rec_emoji = "🌟"
            elif recovery_score >= 45:
                rec_color = "#ffc832"
                rec_label = "Moderate Recovery"
                rec_emoji = "⚡"
            else:
                rec_color = "#ff4b4b"
                rec_label = "Slow Recovery"
                rec_emoji = "⚠️"

            st.markdown(
                f'<div class="recovery-score-box">'
                f'<div style="font-size:2rem;">{rec_emoji}</div>'
                f'<div class="recovery-score-value">{int(recovery_score)}</div>'
                f'<div class="metric-label">Recovery Score</div>'
                f'<div style="color:{rec_color}; font-size:1rem; font-weight:600; margin-top:0.5rem;">'
                f'{rec_label}</div></div>',
                unsafe_allow_html=True,
            )

        with rec_col2:
            st.markdown(
                '<div class="about-card">'
                '<strong style="color:#7b68ee;">📖 How Recovery Score Works</strong><br><br>'
                '<span style="color:#c8cdd6;">'
                'The Recovery Score measures how quickly your body returns to baseline after stress events. '
                'It analyzes the <strong>rate of decline</strong> in stress probability after each high-stress peak.<br><br>'
                '• <strong style="color:#00d4aa;">70–100 (Excellent):</strong> Rapid stress recovery — your body efficiently returns to calm<br>'
                '• <strong style="color:#ffc832;">45–69 (Moderate):</strong> Average recovery — room for improvement in stress management<br>'
                '• <strong style="color:#ff4b4b;">0–44 (Slow):</strong> Prolonged stress periods — consider active relaxation techniques<br><br>'
                '<em style="color:#8892a4;">Tip: Deep breathing, progressive muscle relaxation, and short walks '
                'can significantly improve your recovery speed.</em>'
                '</span></div>',
                unsafe_allow_html=True,
            )

        # ── Stress Zones Breakdown ──
        st.markdown(
            '<div class="analytics-header">🗺️ Stress Zones Breakdown</div>',
            unsafe_allow_html=True,
        )

        high_stress_count = (log_df["stress_prob"] > 60).sum()
        mid_stress_count = ((log_df["stress_prob"] >= 20) & (log_df["stress_prob"] <= 60)).sum()
        low_stress_count = (log_df["stress_prob"] < 20).sum()

        zone_df = pd.DataFrame({
            "Zone": ["🟢 Stress-Free (< 20%)", "🟡 Moderate (20–60%)", "🔴 High Stress (> 60%)"],
            "Count": [low_stress_count, mid_stress_count, high_stress_count],
            "Percentage": [
                f"{low_stress_count/len(log_df)*100:.1f}%",
                f"{mid_stress_count/len(log_df)*100:.1f}%",
                f"{high_stress_count/len(log_df)*100:.1f}%",
            ],
        })

        z1, z2 = st.columns([1, 1])
        with z1:
            fig_zones = go.Figure(go.Pie(
                labels=zone_df["Zone"],
                values=zone_df["Count"],
                hole=0.55,
                marker=dict(colors=["#00d4aa", "#ffc832", "#ff4b4b"]),
                textinfo="percent+label",
                textfont=dict(size=12, color="#fafafa"),
                hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_zones.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                annotations=[dict(text="Zones", x=0.5, y=0.5, font_size=16,
                                  font_color="#fafafa", showarrow=False)],
            )
            st.plotly_chart(fig_zones, use_container_width=True)

        with z2:
            # Physiological averages per zone
            for zone_name, zone_color, condition in [
                ("Stress-Free", "#00d4aa", log_df["stress_prob"] < 20),
                ("Moderate", "#ffc832", (log_df["stress_prob"] >= 20) & (log_df["stress_prob"] <= 60)),
                ("High Stress", "#ff4b4b", log_df["stress_prob"] > 60),
            ]:
                zone_data = log_df[condition]
                if len(zone_data) > 0:
                    st.markdown(
                        f'<div class="about-card" style="padding:0.8rem 1.2rem; border-left:3px solid {zone_color};">'
                        f'<strong style="color:{zone_color};">{zone_name}</strong> — '
                        f'<span style="color:#c8cdd6;">'
                        f'HR: <strong>{zone_data["hr"].mean():.0f}</strong> bpm · '
                        f'RMSSD: <strong>{zone_data["rmssd"].mean():.1f}</strong> ms · '
                        f'SCL: <strong>{zone_data["scl"].mean():.1f}</strong> µS'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )

        # ════════════════════════════════════════════════════════════════
        # SECTION 2: PRE-EMPTIVE STRESS ALERT
        # ════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown(
            '<div class="section-header">🚨 Pre-emptive Stress Alert</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#8892a4;">Analyzing trends to predict potential burnout and provide early warnings.</p>',
            unsafe_allow_html=True,
        )

        # Trend analysis: compare first half vs second half of readings
        n_total = len(log_df)
        split = n_total // 2
        if split >= 2:
            earlier = log_df.iloc[:split]
            recent = log_df.iloc[split:]

            earlier_avg = earlier["stress_prob"].mean()
            recent_avg = recent["stress_prob"].mean()
            change_pct = ((recent_avg - earlier_avg) / max(earlier_avg, 1)) * 100

            # Determine trend
            if change_pct > 10:
                trend_type = "up"
                trend_icon = "📈"
                trend_text = f"Stress levels have <strong>increased {abs(change_pct):.0f}%</strong>"
                badge_class = "trend-up"
                badge_text = f"↑ +{abs(change_pct):.0f}%"
            elif change_pct < -10:
                trend_type = "down"
                trend_icon = "📉"
                trend_text = f"Stress levels have <strong>decreased {abs(change_pct):.0f}%</strong>"
                badge_class = "trend-down"
                badge_text = f"↓ -{abs(change_pct):.0f}%"
            else:
                trend_type = "stable"
                trend_icon = "➡️"
                trend_text = "Stress levels are <strong>relatively stable</strong>"
                badge_class = "trend-stable"
                badge_text = f"~ {abs(change_pct):.0f}%"

            # Trend badge
            st.markdown(
                f'<div style="text-align:center; margin:1rem 0;">'
                f'<span class="trend-badge {badge_class}">{trend_icon} {badge_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Alert Banner
            if trend_type == "up":
                if change_pct > 30:
                    alert_class = "alert-danger"
                    alert_msg = (
                        f'🚨 <strong>Burnout Warning:</strong> Your average stress level has surged '
                        f'<strong>{abs(change_pct):.0f}%</strong> over your recent readings '
                        f'(from {earlier_avg:.0f}% → {recent_avg:.0f}%). '
                        f'<strong>Strongly consider taking a break and consulting a mentor or counselor.</strong>'
                    )
                else:
                    alert_class = "alert-warning"
                    alert_msg = (
                        f'⚠️ <strong>Stress Increasing:</strong> Your average stress level has increased '
                        f'<strong>{abs(change_pct):.0f}%</strong> over your recent readings '
                        f'(from {earlier_avg:.0f}% → {recent_avg:.0f}%). '
                        f'Consider taking a break and incorporating relaxation techniques.'
                    )
            elif trend_type == "down":
                alert_class = "alert-success"
                alert_msg = (
                    f'✅ <strong>Improving!</strong> Your stress level has decreased by '
                    f'<strong>{abs(change_pct):.0f}%</strong> recently '
                    f'(from {earlier_avg:.0f}% → {recent_avg:.0f}%). '
                    f'Keep up the good work! Your stress management is paying off.'
                )
            else:
                alert_class = "alert-warning"
                alert_msg = (
                    f'➡️ <strong>Stable:</strong> Your stress levels have remained steady '
                    f'around <strong>{recent_avg:.0f}%</strong>. '
                    + (f'This is within a healthy range. Maintain your current habits.'
                       if recent_avg < 45 else
                       f'However, at {recent_avg:.0f}% this is elevated. Try active relaxation to lower your baseline.')
                )

            st.markdown(
                f'<div class="alert-banner {alert_class}">'
                f'<span style="color:#c8cdd6;">{alert_msg}</span></div>',
                unsafe_allow_html=True,
            )

            # ── Trend Comparison Chart ──
            st.markdown(
                '<div class="analytics-header">📊 Trend Comparison</div>',
                unsafe_allow_html=True,
            )

            fig_trend = go.Figure()

            # Earlier period
            fig_trend.add_trace(go.Scatter(
                x=earlier["entry_num"],
                y=earlier["stress_prob"],
                mode="lines+markers",
                name=f"Earlier (avg {earlier_avg:.0f}%)",
                line=dict(color="#00b4d8", width=2),
                marker=dict(size=6),
            ))

            # Recent period
            fig_trend.add_trace(go.Scatter(
                x=recent["entry_num"],
                y=recent["stress_prob"],
                mode="lines+markers",
                name=f"Recent (avg {recent_avg:.0f}%)",
                line=dict(color="#ff4b4b" if trend_type == "up" else "#00d4aa", width=2),
                marker=dict(size=6),
            ))

            # Add average lines
            fig_trend.add_hline(y=earlier_avg, line_dash="dot", line_color="#00b4d8",
                                annotation_text=f"Earlier avg: {earlier_avg:.0f}%",
                                annotation_font_color="#00b4d8")
            fig_trend.add_hline(y=recent_avg, line_dash="dot",
                                line_color="#ff4b4b" if trend_type == "up" else "#00d4aa",
                                annotation_text=f"Recent avg: {recent_avg:.0f}%",
                                annotation_font_color="#ff4b4b" if trend_type == "up" else "#00d4aa")

            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(
                    title="Reading #", gridcolor="rgba(255,255,255,0.05)",
                    tickfont=dict(color="#8892a4"), title_font=dict(color="#8892a4"),
                ),
                yaxis=dict(
                    title="Stress Probability (%)", range=[0, 100],
                    gridcolor="rgba(255,255,255,0.05)",
                    tickfont=dict(color="#8892a4"), title_font=dict(color="#8892a4"),
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(color="#fafafa", size=12),
                ),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

            # ── Physiological Trend Cards ──
            st.markdown(
                '<div class="analytics-header">🩺 Physiological Trend Details</div>',
                unsafe_allow_html=True,
            )

            trend_col1, trend_col2, trend_col3 = st.columns(3)

            for col, (metric, label, emoji, unit) in zip(
                [trend_col1, trend_col2, trend_col3],
                [("hr", "Heart Rate", "💓", "bpm"), ("rmssd", "HRV (RMSSD)", "📈", "ms"), ("scl", "Skin Conductance", "⚡", "µS")]
            ):
                with col:
                    earlier_val = earlier[metric].mean()
                    recent_val = recent[metric].mean()
                    diff = recent_val - earlier_val
                    diff_pct = (diff / max(earlier_val, 0.01)) * 100

                    if metric == "rmssd":
                        # For RMSSD, decrease is bad (more stress)
                        diff_color = "#ff4b4b" if diff < 0 else "#00d4aa"
                    else:
                        # For HR and SCL, increase is bad
                        diff_color = "#ff4b4b" if diff > 0 else "#00d4aa"

                    diff_sign = "+" if diff > 0 else ""
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div style="font-size:1.1rem;">{emoji} <strong>{label}</strong></div>'
                        f'<div style="margin:0.5rem 0;">'
                        f'<span style="color:#8892a4;">Before:</span> <strong>{earlier_val:.1f}</strong> {unit}<br>'
                        f'<span style="color:#8892a4;">Now:</span> <strong>{recent_val:.1f}</strong> {unit}'
                        f'</div>'
                        f'<div style="color:{diff_color}; font-weight:700; font-size:1.1rem;">'
                        f'{diff_sign}{diff:.1f} {unit} ({diff_sign}{diff_pct:.0f}%)</div></div>',
                        unsafe_allow_html=True,
                    )

            # ── Personalized Recommendations ──
            if trend_type == "up":
                st.markdown("")
                st.markdown(
                    '<div class="analytics-header">💡 Personalized Recommendations</div>',
                    unsafe_allow_html=True,
                )

                rec_c1, rec_c2 = st.columns(2)
                with rec_c1:
                    st.markdown(
                        '<div class="about-card" style="border-left:3px solid #ff6b6b;">'
                        '<strong style="color:#ff6b6b;">🫁 Immediate Actions</strong><br><br>'
                        '<span style="color:#c8cdd6;">'
                        '• Take a <strong>15-minute break</strong> every 2 hours<br>'
                        '• Practice <strong>box breathing</strong> (4-4-4-4 seconds) 3x daily<br>'
                        '• Go for a <strong>10-minute walk</strong> outside — nature reduces cortisol 12-16%<br>'
                        '• Drink water and avoid caffeine for the next few hours'
                        '</span></div>',
                        unsafe_allow_html=True,
                    )
                with rec_c2:
                    st.markdown(
                        '<div class="about-card" style="border-left:3px solid #7b68ee;">'
                        '<strong style="color:#7b68ee;">🌙 This Week\'s Plan</strong><br><br>'
                        '<span style="color:#c8cdd6;">'
                        '• Prioritize <strong>7-9 hours</strong> of sleep tonight<br>'
                        '• Schedule <strong>2-3 social activities</strong> — social support buffers stress<br>'
                        '• Try <strong>journaling</strong> for 10 minutes before bed<br>'
                        '• Consider reducing workload or <strong>delegating tasks</strong> if possible'
                        '</span></div>',
                        unsafe_allow_html=True,
                    )

        else:
            st.info("📊 Need at least 4 readings to compute trends. Log more data or use sample data.")

        # ── Sleep Quality Impact Predictor ─────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div class="section-header">😴 Sleep Quality Impact Predictor</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#8892a4;">'
            'Map your daily stress levels against sleep to see how high-stress days '
            'impact total sleep time and deep sleep quality.'
            '</p>',
            unsafe_allow_html=True,
        )

        # Load sleep data CSV from Final_CSVs/sleep_data.csv (if present)
        # If missing or dates don't overlap with stress data, fall back to a realistic
        # simulated sleep pattern derived from the stress history so the widget always works.
        sleep_path = os.path.join(CSV_DIR, "sleep_data.csv")
        sleep_df = None

        if os.path.exists(sleep_path):
            try:
                raw_sleep_df = pd.read_csv(sleep_path)
                required_cols = {"date", "total_sleep_hours", "deep_sleep_percent"}
                if required_cols.issubset(set(raw_sleep_df.columns)):
                    sleep_df = raw_sleep_df.copy()
                    sleep_df["date"] = pd.to_datetime(sleep_df["date"]).dt.date
                else:
                    missing = required_cols.difference(set(raw_sleep_df.columns))
                    st.warning(
                        f"`sleep_data.csv` is missing required columns: {', '.join(sorted(missing))}. "
                        "Falling back to simulated sleep data based on stress levels."
                    )
            except Exception as e:
                st.warning(f"Unable to read `sleep_data.csv` ({e}). Falling back to simulated sleep data.")

        # Aggregate stress per date
        daily_stress = (
            log_df.groupby("date")
            .agg(
                avg_stress=("stress_prob", "mean"),
                max_stress=("stress_prob", "max"),
                high_stress_count=("stress_prob", lambda s: (s > 60).sum()),
            )
            .reset_index()
        )

        simulated_sleep = False

        if sleep_df is None:
            # No valid sleep file: simulate sleep metrics from stress pattern
            sim = daily_stress[["date", "avg_stress"]].copy()
            # Higher stress → shorter and lighter sleep
            sim["total_sleep_hours"] = np.clip(
                8.0 - (sim["avg_stress"] / 100.0) * 3.0 + np.random.normal(0, 0.3, len(sim)),
                4.0,
                9.0,
            )
            sim["deep_sleep_percent"] = np.clip(
                25.0 - (sim["avg_stress"] / 100.0) * 10.0 + np.random.normal(0, 2.0, len(sim)),
                5.0,
                40.0,
            )
            sleep_df = sim
            simulated_sleep = True

        merged = pd.merge(daily_stress, sleep_df, on="date", how="inner")

        # After merging, ensure we have a single `avg_stress` column
        if "avg_stress_x" in merged.columns:
            merged["avg_stress"] = merged["avg_stress_x"]
            merged = merged.drop(columns=[c for c in ["avg_stress_x", "avg_stress_y"] if c in merged.columns])

        if merged.empty:
            # If user-provided file has dates that don't match the stress history,
            # regenerate a simulated sleep profile directly from daily_stress.
            sim = daily_stress[["date", "avg_stress"]].copy()
            sim["total_sleep_hours"] = np.clip(
                8.0 - (sim["avg_stress"] / 100.0) * 3.0 + np.random.normal(0, 0.3, len(sim)),
                4.0,
                9.0,
            )
            sim["deep_sleep_percent"] = np.clip(
                25.0 - (sim["avg_stress"] / 100.0) * 10.0 + np.random.normal(0, 2.0, len(sim)),
                5.0,
                40.0,
            )
            sleep_df = sim
            merged = pd.merge(daily_stress, sleep_df, on="date", how="inner")
            if "avg_stress_x" in merged.columns:
                merged["avg_stress"] = merged["avg_stress_x"]
                merged = merged.drop(columns=[c for c in ["avg_stress_x", "avg_stress_y"] if c in merged.columns])
            simulated_sleep = True

        if merged.empty:
            st.info("Not enough data to compute sleep impact yet. Log a few days of readings.")
        else:
                        # Optional: load predictive sleep model, if trained
                        sleep_model_path = os.path.join(MODEL_DIR, "sleep_model.pkl")
                        sleep_scaler_path = os.path.join(MODEL_DIR, "sleep_scaler.pkl")
                        sleep_metadata_path = os.path.join(MODEL_DIR, "sleep_metadata.pkl")

                        sleep_model = sleep_scaler = sleep_meta = None
                        if os.path.exists(sleep_model_path) and os.path.exists(sleep_scaler_path):
                            try:
                                @st.cache_resource
                                def load_sleep_model():
                                    model = joblib.load(sleep_model_path)
                                    scaler = joblib.load(sleep_scaler_path)
                                    metadata = joblib.load(sleep_metadata_path) if os.path.exists(sleep_metadata_path) else {}
                                    return model, scaler, metadata

                                sleep_model, sleep_scaler, sleep_meta = load_sleep_model()
                            except Exception as e:
                                st.warning(f"Sleep prediction model could not be loaded: {e}")

                        # If model is available, generate predicted sleep for comparison
                        if sleep_model is not None and sleep_scaler is not None:
                            # Sort by date and create previous-night features
                            merged = merged.sort_values("date").reset_index(drop=True)
                            merged["prev_total_sleep_hours"] = merged["total_sleep_hours"].shift(1).bfill()
                            merged["prev_deep_sleep_percent"] = merged["deep_sleep_percent"].shift(1).bfill()
                            merged["prev_sleep_efficiency"] = merged.get("sleep_efficiency", pd.Series(index=merged.index, data=90)).shift(1).bfill()

                            feature_cols = sleep_meta.get(
                                "feature_cols",
                                [
                                    "avg_stress",
                                    "max_stress",
                                    "high_stress_count",
                                    "hr_mean",
                                    "rmssd_mean",
                                    "scl_mean",
                                    "prev_total_sleep_hours",
                                    "prev_deep_sleep_percent",
                                    "prev_sleep_efficiency",
                                ],
                            )

                            # Some physiological means might not exist; fall back to proxies
                            for col in ["hr_mean", "rmssd_mean", "scl_mean"]:
                                if col not in merged.columns:
                                    if col == "hr_mean":
                                        merged[col] = log_df.groupby("date")["hr"].transform("mean")
                                    elif col == "rmssd_mean":
                                        merged[col] = log_df.groupby("date")["rmssd"].transform("mean")
                                    elif col == "scl_mean":
                                        merged[col] = log_df.groupby("date")["scl"].transform("mean")

                            feature_df = merged[feature_cols].copy()
                            X_sleep = np.nan_to_num(feature_df.values, nan=0.0, posinf=0.0, neginf=0.0)
                            X_sleep_scaled = sleep_scaler.transform(X_sleep)
                            preds = sleep_model.predict(X_sleep_scaled)

                            merged["pred_total_sleep_hours"] = preds[:, 0]
                            merged["pred_deep_sleep_percent"] = preds[:, 1]

                        # If we're using simulated sleep data, let the user know
                        if simulated_sleep:
                            st.info(
                                "Using simulated sleep data based on your stress levels. "
                                "Add or update `Final_CSVs/sleep_data.csv` with real tracker data "
                                "to see your actual sleep patterns here."
                            )

                        # ── Summary cards: low- vs high-stress sleep ──
                        low_stress_days = merged[merged["avg_stress"] <= 40]
                        high_stress_days = merged[merged["avg_stress"] >= 60]

                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.markdown(
                                '<div class="metric-card">'
                                f'<div class="metric-value" style="font-size:1.6rem;">{len(merged)}</div>'
                                '<div class="metric-label">Days with Sleep & Stress Data</div></div>',
                                unsafe_allow_html=True,
                            )
                        with sc2:
                            if len(low_stress_days) > 0:
                                avg_sleep_low = low_stress_days["total_sleep_hours"].mean()
                                st.markdown(
                                    '<div class="metric-card">'
                                    f'<div class="metric-value" style="font-size:1.6rem;">{avg_sleep_low:.1f}h</div>'
                                    '<div class="metric-label">Sleep on Low-Stress Days (≤ 40%)</div></div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="metric-card">'
                                    '<div class="metric-value" style="font-size:1.2rem;">—</div>'
                                    '<div class="metric-label">No Low-Stress Days Logged</div></div>',
                                    unsafe_allow_html=True,
                                )
                        with sc3:
                            if len(high_stress_days) > 0:
                                avg_sleep_high = high_stress_days["total_sleep_hours"].mean()
                                st.markdown(
                                    '<div class="metric-card">'
                                    f'<div class="metric-value" style="font-size:1.6rem;">{avg_sleep_high:.1f}h</div>'
                                    '<div class="metric-label">Sleep on High-Stress Days (≥ 60%)</div></div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<div class="metric-card">'
                                    '<div class="metric-value" style="font-size:1.2rem;">—</div>'
                                    '<div class="metric-label">No High-Stress Days Logged</div></div>',
                                    unsafe_allow_html=True,
                                )

                        st.markdown("")

                        # ── Correlation charts ──
                        c1, c2 = st.columns(2)

                        with c1:
                            fig_sleep_hours = go.Figure()
                            fig_sleep_hours.add_trace(
                                go.Scatter(
                                    x=merged["avg_stress"],
                                    y=merged["total_sleep_hours"],
                                    mode="markers+lines",
                                    line=dict(color="#00b4d8", width=2),
                                    marker=dict(size=8, color="#00d4aa"),
                                    hovertemplate="Stress: %{x:.1f}%<br>Sleep: %{y:.2f}h<extra></extra>",
                                    name="Actual Sleep",
                                )
                            )
                            if "pred_total_sleep_hours" in merged.columns:
                                fig_sleep_hours.add_trace(
                                    go.Scatter(
                                        x=merged["avg_stress"],
                                        y=merged["pred_total_sleep_hours"],
                                        mode="lines",
                                        line=dict(color="#ff4b4b", width=2, dash="dash"),
                                        name="Predicted Sleep",
                                        hovertemplate="Stress: %{x:.1f}%<br>Pred Sleep: %{y:.2f}h<extra></extra>",
                                    )
                                )
                            fig_sleep_hours.update_layout(
                                title="Daily Stress vs Total Sleep Hours",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=320,
                                margin=dict(l=10, r=10, t=40, b=40),
                                xaxis=dict(
                                    title="Average Daily Stress (%)",
                                    range=[0, 100],
                                    gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#8892a4"),
                                    title_font=dict(color="#8892a4"),
                                ),
                                yaxis=dict(
                                    title="Total Sleep (hours)",
                                    gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#8892a4"),
                                    title_font=dict(color="#8892a4"),
                                ),
                            )
                            st.plotly_chart(fig_sleep_hours, use_container_width=True)

                        with c2:
                            fig_deep = go.Figure()
                            fig_deep.add_trace(
                                go.Scatter(
                                    x=merged["avg_stress"],
                                    y=merged["deep_sleep_percent"],
                                    mode="markers+lines",
                                    line=dict(color="#7b68ee", width=2),
                                    marker=dict(size=8, color="#ffc832"),
                                    hovertemplate="Stress: %{x:.1f}%<br>Deep Sleep: %{y:.1f}%<extra></extra>",
                                    name="Actual Deep Sleep",
                                )
                            )
                            if "pred_deep_sleep_percent" in merged.columns:
                                fig_deep.add_trace(
                                    go.Scatter(
                                        x=merged["avg_stress"],
                                        y=merged["pred_deep_sleep_percent"],
                                        mode="lines",
                                        line=dict(color="#00d4aa", width=2, dash="dash"),
                                        name="Predicted Deep Sleep",
                                        hovertemplate="Stress: %{x:.1f}%<br>Pred Deep: %{y:.1f}%<extra></extra>",
                                    )
                                )
                            fig_deep.update_layout(
                                title="Daily Stress vs Deep Sleep %",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=320,
                                margin=dict(l=10, r=10, t=40, b=40),
                                xaxis=dict(
                                    title="Average Daily Stress (%)",
                                    range=[0, 100],
                                    gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#8892a4"),
                                    title_font=dict(color="#8892a4"),
                                ),
                                yaxis=dict(
                                    title="Deep Sleep (%)",
                                    range=[0, 100],
                                    gridcolor="rgba(255,255,255,0.05)",
                                    tickfont=dict(color="#8892a4"),
                                    title_font=dict(color="#8892a4"),
                                ),
                            )
                            st.plotly_chart(fig_deep, use_container_width=True)

                        # ── Textual insight comparing low vs high stress days ──
                        if len(low_stress_days) > 0 and len(high_stress_days) > 0:
                            avg_sleep_low = low_stress_days["total_sleep_hours"].mean()
                            avg_sleep_high = high_stress_days["total_sleep_hours"].mean()
                            avg_deep_low = low_stress_days["deep_sleep_percent"].mean()
                            avg_deep_high = high_stress_days["deep_sleep_percent"].mean()

                            diff_sleep_hours = avg_sleep_low - avg_sleep_high
                            diff_deep = avg_deep_low - avg_deep_high

                            insight_text = (
                                '<div class="about-card">'
                                '<strong style="color:#7b68ee;">📌 Key Sleep Insight</strong><br><br>'
                                f'<span style="color:#c8cdd6;">'
                                f'On <strong>low-stress days (≤ 40%)</strong>, you average '
                                f'<strong>{avg_sleep_low:.1f} hours</strong> of sleep with '
                                f'<strong>{avg_deep_low:.0f}%</strong> deep sleep.<br>'
                                f'On <strong>high-stress days (≥ 60%)</strong>, you average '
                                f'<strong>{avg_sleep_high:.1f} hours</strong> of sleep with '
                                f'<strong>{avg_deep_high:.0f}%</strong> deep sleep.<br><br>'
                            )

                            if "pred_total_sleep_hours" in merged.columns and "pred_deep_sleep_percent" in merged.columns:
                                mae_sleep = np.mean(
                                    np.abs(
                                        merged["total_sleep_hours"] - merged["pred_total_sleep_hours"]
                                    )
                                )
                                mae_deep = np.mean(
                                    np.abs(
                                        merged["deep_sleep_percent"] - merged["pred_deep_sleep_percent"]
                                    )
                                )
                                insight_text += (
                                    f'Compared to the model\'s expectations, your actual sleep differs by '
                                    f'about <strong>{mae_sleep:.1f} hours</strong> and '
                                    f'<strong>{mae_deep:.0f}%</strong> deep sleep on average.<br><br>'
                                )

                            insight_text += (
                                f'That is roughly '
                                f'<strong>{abs(diff_sleep_hours):.1f} hours</strong> less sleep and '
                                f'<strong>{abs(diff_deep):.0f}%</strong> less deep sleep on high-stress days.'
                                '</span></div>'
                            )

                            st.markdown(insight_text, unsafe_allow_html=True)

                        # ── Correlation-based summary: how strongly stress impacts sleep ──
                        if len(merged) >= 3:
                            corr_sleep = merged["avg_stress"].corr(merged["total_sleep_hours"])
                            corr_deep = merged["avg_stress"].corr(merged["deep_sleep_percent"])

                            def describe_correlation(value: float) -> str:
                                if pd.isna(value):
                                    return "no clear relationship"
                                abs_v = abs(value)
                                if abs_v < 0.2:
                                    return "very weak relationship"
                                if abs_v < 0.4:
                                    level = "weak"
                                elif abs_v < 0.6:
                                    level = "moderate"
                                else:
                                    level = "strong"
                                direction = "negative" if value < 0 else "positive"
                                return f"{level} {direction} relationship"

                            sleep_desc = describe_correlation(corr_sleep)
                            deep_desc = describe_correlation(corr_deep)

                            st.markdown(
                                '<div class="about-card" style="margin-top:0.5rem;">'
                                '<strong style="color:#7b68ee;">🧠 How Stress Is Affecting Your Sleep</strong><br><br>'
                                f'<span style="color:#c8cdd6;">'
                                f'Across the days with data, there is a <strong>{sleep_desc}</strong> between '
                                f'<strong>daily stress</strong> and <strong>total sleep hours</strong> '
                                f'(correlation ≈ {corr_sleep:.2f} if available).<br>'
                                f'There is a <strong>{deep_desc}</strong> between '
                                f'<strong>daily stress</strong> and <strong>deep sleep %</strong> '
                                f'(correlation ≈ {corr_deep:.2f} if available).<br><br>'
                                'Negative relationships mean that on higher-stress days you tend to sleep less '
                                'or get less deep sleep; positive relationships mean the opposite.</span></div>',
                                unsafe_allow_html=True,
                            )

        # ── Session Data Table ──
        st.markdown("---")
        with st.expander("📋 View All Logged Readings", expanded=False):
            display_df = log_df[["timestamp", "hr", "rmssd", "scl", "stress_prob", "prediction"]].copy()
            display_df.columns = ["Timestamp", "HR (bpm)", "RMSSD (ms)", "SCL (µS)", "Stress %", "Stressed?"]
            display_df["Stressed?"] = display_df["Stressed?"].map({0: "😊 No", 1: "😰 Yes"})
            display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(display_df, use_container_width=True, hide_index=True, height=300)

        # Disclaimer
        st.markdown(
            '<div style="background:rgba(255,200,50,0.08); border:1px solid rgba(255,200,50,0.25); '
            'border-radius:12px; padding:1rem; text-align:center; margin-top:1rem;">'
            '<span style="color:#ffc832;">⚠️</span> '
            '<span style="color:#c8cdd6; font-size:0.85rem;">'
            'Stress logs and sleep data are stored locally in CSV files inside this project folder. '
            'This dashboard is for wellness insights only and is not a substitute for professional health monitoring.'
            '</span></div>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📖 About":
    render_topbar("About")
    st.markdown('<h1 class="main-title">About</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Methodology and approach details</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-header">🔬 Five Approaches to Stress Detection</div>', unsafe_allow_html=True)

    approaches = [
        (
            "1️⃣ Generic Models",
            "A single **person-independent model** trained on all available data. "
            "Uses algorithms like Random Forest, Gradient Boosting, SVM, and Logistic Regression. "
            "Two variants: with and without participant ID as a feature.",
        ),
        (
            "2️⃣ User-Based Models",
            "**Personalized models** built separately for each individual. "
            "ML approach trains per-user models; DL approach uses **Multitask Learning (MTL)** "
            "where each user's stress detection is a separate task with shared hidden layers.",
        ),
        (
            "3️⃣ Single-Attribute (Personality) Models",
            "Users are **grouped by personality** using K-means clustering. "
            "Number of clusters (k) is determined by Elbow Method and Silhouette Score. "
            "A separate model is trained for each personality group.",
        ),
        (
            "4️⃣ Multi-Attribute Models ⭐",
            "**Best-performing approach.** Groups users by multiple attributes: "
            "caffeine intake, exercise, sleep, age, gender, plus physiological stats (mean, min, std). "
            "PCA is used when features are high-dimensional. Achieves up to **0.9960 F1-score**.",
        ),
        (
            "5️⃣ Fuzzy Models",
            "Uses **Fuzzy C-Means** clustering where users belong to all clusters with "
            "varying membership degrees. Data is split proportionally across clusters. "
            "Addresses the limitation of hard cluster assignments.",
        ),
    ]

    for title, desc in approaches:
        st.markdown(
            f'<div class="about-card">'
            f'<strong style="font-size:1.1rem;">{title}</strong><br><br>'
            f'<span style="color:#c8cdd6;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">📊 Evaluation</div>', unsafe_allow_html=True)
    st.markdown("""
    - **Primary Metric:** F1-Score (weighted) — chosen because datasets ADARP and LifeSnaps are imbalanced
    - **ML Algorithms:** LDA, SVM, Logistic Regression, Decision Tree, Naive Bayes, KNN, QDA, 
      Random Forest, Extra Trees, AdaBoost, Gradient Boosting, LightGBM
    - **DL Framework:** TensorFlow — MTL with shared hidden layers and task-specific output layers
    - **Validation:** Train/Test split with cross-validation
    """)

    st.markdown('<div class="section-header">📐 Physiological Features</div>', unsafe_allow_html=True)
    features_info = {
        "Feature": ["Heart Rate (HR)", "HRV / RMSSD", "Skin Conductance (SCL)", "Body Temperature", "Acceleration", "Sleep Metrics", "Physical Activity"],
        "Description": [
            "Beats per minute — increases under stress",
            "Root Mean Square of Successive Differences — decreases under stress",
            "Electrodermal activity — increases with arousal/stress",
            "Peripheral body temperature via wearable",
            "3-axis accelerometer data for movement detection",
            "Duration, efficiency, deep/light/REM ratios",
            "Steps, active minutes, sedentary time",
        ],
        "Datasets": [
            "All four",
            "SWELL, WESAD",
            "All four",
            "WESAD, ADARP",
            "WESAD, ADARP",
            "LifeSnaps",
            "LifeSnaps",
        ],
    }
    st.dataframe(pd.DataFrame(features_info), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:#8892a4; font-size:0.85rem;">'
        'Based on the research: <strong>Personalized Machine Learning Benchmarking for Stress Detection</strong>'
        '</p>',
        unsafe_allow_html=True,
    )
