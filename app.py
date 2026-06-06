import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json

from auth import AuthManager
from data_generator import SyntheticDataGenerator
from models import AHFPredictionModels
from database import DatabaseManager
from notifications import NotificationManager
from explainability import ExplainabilityManager
from monitoring import ModelMonitor
from reporting import ReportGenerator
from data_validation import DataValidator
from alert_system import AlertSystem
from cnn_model import MedicalImageCNN

st.set_page_config(
    page_title="CardioGuard AI — AHF Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="auto"
)

# ─── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

/* ── Responsive container ── */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.75rem !important;
    }
}
@media (min-width: 769px) and (max-width: 1024px) {
    .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }
}

/* ── Page background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%) !important;
    min-height: 100vh;
    color: #e0e7ef !important;
}
body {
    background: #0d1b2a !important;
    color: #e0e7ef !important;
}
/* Default text color for all Streamlit text */
p, span, label, div, h1, h2, h3, h4, h5, h6 {
    color: #cbd5e1;
}
/* Widget labels */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stMultiSelect label, .stSlider label, .stCheckbox label,
.stFileUploader label, .stTextArea label, .stDateInput label,
.stRadio label { color: #90a4ae !important; }
/* Widget values and inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    color: #e0e7ef !important;
    border: 1px solid rgba(30,136,229,0.3) !important;
    border-radius: 10px !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(30,136,229,0.3) !important;
    color: #e0e7ef !important;
    border-radius: 10px !important;
}
/* Metric values */
[data-testid="stMetricValue"] { color: #64b5f6 !important; }
[data-testid="stMetricLabel"] { color: #78909c !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #112240 60%, #0d1b2a 100%);
    border-right: 1px solid rgba(30,136,229,0.25);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stButton > button {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(30,136,229,0.5) !important;
}

/* ── Card component ── */
.card {
    background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(30,136,229,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}
.card:hover {
    border-color: rgba(30,136,229,0.45);
    box-shadow: 0 12px 40px rgba(30,136,229,0.15);
    transform: translateY(-2px);
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #1565c0 50%, #0288d1 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 40px rgba(30,136,229,0.3);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.hero-subtitle {
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    margin: 0.4rem 0 0 0;
    font-weight: 400;
}

/* ── Section header ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #90caf9;
    border-left: 4px solid #1e88e5;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── KPI metric cards ── */
.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
    border: 1px solid rgba(30,136,229,0.25);
    border-radius: 14px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(30,136,229,0.2); }
.kpi-value { font-size: 2rem; font-weight: 800; color: #64b5f6; }
.kpi-label { font-size: 0.8rem; color: #90a4ae; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.25rem; }
.kpi-delta { font-size: 0.75rem; margin-top: 0.2rem; }

/* ── Risk badge ── */
.risk-high { background: linear-gradient(135deg, #c62828, #e53935); color: #fff; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 700; display: inline-block; }
.risk-medium { background: linear-gradient(135deg, #e65100, #fb8c00); color: #fff; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 700; display: inline-block; }
.risk-low { background: linear-gradient(135deg, #1b5e20, #43a047); color: #fff; padding: 0.4rem 1rem; border-radius: 20px; font-weight: 700; display: inline-block; }

/* ── Upload zone ── */
.upload-zone {
    border: 2px dashed rgba(30,136,229,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: rgba(30,136,229,0.05);
    transition: all 0.3s ease;
}
.upload-zone:hover { border-color: #1e88e5; background: rgba(30,136,229,0.1); }

/* ── Finding badge ── */
.finding-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0.2rem;
}
.finding-normal { background: rgba(67,160,71,0.2); color: #81c784; border: 1px solid rgba(67,160,71,0.3); }
.finding-abnormal { background: rgba(229,57,53,0.2); color: #ef9a9a; border: 1px solid rgba(229,57,53,0.3); }
.finding-warning { background: rgba(251,140,0,0.2); color: #ffcc80; border: 1px solid rgba(251,140,0,0.3); }

/* ── Recommendation item ── */
.rec-item {
    background: rgba(30,136,229,0.08);
    border-left: 3px solid #1e88e5;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    color: #b0bec5;
    font-size: 0.9rem;
}
.rec-item.urgent {
    background: rgba(229,57,53,0.1);
    border-left-color: #e53935;
    color: #ef9a9a;
    font-weight: 600;
}

/* ── Progress bar custom ── */
.prog-bar-wrap { background: rgba(255,255,255,0.08); border-radius: 10px; height: 10px; overflow: hidden; margin: 0.3rem 0; }
.prog-bar-fill { height: 100%; border-radius: 10px; transition: width 0.8s ease; }

/* ── Login card ── */
.login-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(30,136,229,0.3);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}

/* ── Streamlit widget overrides ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(30,136,229,0.3) !important;
    border-radius: 10px !important;
    color: #e0e7ef !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1e88e5 !important;
    box-shadow: 0 0 0 3px rgba(30,136,229,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30,136,229,0.45) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00acc1 0%, #00838f 100%) !important;
}

.stForm { border: none !important; }
[data-testid="stForm"] { background: transparent !important; border: none !important; padding: 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #90a4ae;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e88e5, #1565c0) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Checkbox ── */
.stCheckbox label { color: #b0bec5 !important; }

/* ── Divider ── */
hr { border-color: rgba(30,136,229,0.15) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); }
::-webkit-scrollbar-thumb { background: rgba(30,136,229,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #1e88e5; }

/* ── Nav pill in sidebar ── */
.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 10px;
    margin: 2px 0;
    cursor: pointer;
    transition: all 0.2s;
    color: #90a4ae;
    font-weight: 500;
}
.nav-item.active {
    background: linear-gradient(135deg, rgba(30,136,229,0.25), rgba(21,101,192,0.15));
    color: #64b5f6;
    border-left: 3px solid #1e88e5;
}

/* ── Spinner text ── */
.stSpinner > div { color: #64b5f6 !important; }

/* ── Alert / info boxes ── */
.stAlert { border-radius: 12px !important; }

/* Image display */
.img-display {
    border-radius: 12px;
    border: 1px solid rgba(30,136,229,0.3);
    overflow: hidden;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #43a047; box-shadow: 0 0 6px #43a047; }
.status-dot.red { background: #e53935; box-shadow: 0 0 6px #e53935; }
.status-dot.orange { background: #fb8c00; box-shadow: 0 0 6px #fb8c00; }

/* ═══════════════════════════════════════
   RESPONSIVE DESIGN — ALL BREAKPOINTS
   ═══════════════════════════════════════ */

/* ── Mobile (≤ 768px) ── */
@media (max-width: 768px) {

    /* Hero banner */
    .hero-banner {
        padding: 1.1rem 1rem !important;
        border-radius: 14px !important;
        margin-bottom: 0.9rem !important;
    }
    .hero-banner::before { display: none; }
    .hero-title { font-size: 1.2rem !important; line-height: 1.3 !important; }
    .hero-subtitle { font-size: 0.78rem !important; margin-top: 0.2rem !important; }

    /* Cards */
    .card { padding: 0.9rem !important; border-radius: 12px !important; margin-bottom: 0.65rem !important; }
    .login-card { padding: 1.1rem !important; border-radius: 14px !important; }

    /* KPI */
    .kpi-card { padding: 0.8rem !important; }
    .kpi-value { font-size: 1.45rem !important; }
    .kpi-label { font-size: 0.68rem !important; }

    /* Section header */
    .section-header { font-size: 1rem !important; margin: 0.9rem 0 0.6rem 0 !important; }

    /* Badges */
    .risk-high, .risk-medium, .risk-low {
        padding: 0.28rem 0.65rem !important;
        font-size: 0.82rem !important;
    }
    .finding-badge { font-size: 0.72rem !important; padding: 0.22rem 0.55rem !important; }

    /* Upload zone */
    .upload-zone { padding: 1.1rem 0.75rem !important; }

    /* Rec items */
    .rec-item { font-size: 0.78rem !important; padding: 0.45rem 0.7rem !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        padding: 0.38rem 0.55rem !important;
        font-size: 0.78rem !important;
    }

    /* Columns — allow wrapping so forms don't overflow */
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 140px !important;
        flex: 1 1 140px !important;
    }

    /* Buttons — touch-comfortable */
    .stButton > button {
        padding: 0.55rem 0.9rem !important;
        font-size: 0.88rem !important;
        min-height: 42px !important;
    }

    /* Inputs — touch-comfortable */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        font-size: 1rem !important;
        min-height: 42px !important;
    }
    .stSelectbox > div > div { min-height: 42px !important; font-size: 1rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { min-width: 230px !important; max-width: 270px !important; }

    /* Dataframe horizontal scroll */
    .stDataFrame, [data-testid="stDataFrame"] { overflow-x: auto !important; }

    /* Progress bar */
    .prog-bar-wrap { height: 8px !important; }

    /* Plotly full width */
    .js-plotly-plot, .plotly { width: 100% !important; }
}

/* ── Tablet (769px – 1024px) ── */
@media (min-width: 769px) and (max-width: 1024px) {

    .hero-banner { padding: 1.5rem 1.6rem !important; border-radius: 16px !important; }
    .hero-title { font-size: 1.6rem !important; }
    .hero-subtitle { font-size: 0.88rem !important; }

    .card { padding: 1.2rem !important; }
    .kpi-value { font-size: 1.75rem !important; }
    .kpi-label { font-size: 0.74rem !important; }

    [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 200px !important;
        flex: 1 1 200px !important;
    }

    .stTabs [data-baseweb="tab"] { padding: 0.44rem 0.85rem !important; font-size: 0.88rem !important; }
}

/* ── Large desktop (≥ 1400px) ── */
@media (min-width: 1400px) {
    .hero-title { font-size: 2.3rem !important; }
    .hero-subtitle { font-size: 1.05rem !important; }
    .kpi-value { font-size: 2.2rem !important; }
    .card { padding: 1.75rem !important; }
}

/* ── Touch-only devices (no hover hardware) ── */
@media (hover: none) and (pointer: coarse) {
    .stButton > button { min-height: 46px !important; }
    .stCheckbox label { font-size: 1rem !important; line-height: 1.6 !important; }
    .stSlider { padding: 10px 0 !important; }
    /* Remove sticky hover effects on touch */
    .card:hover, .kpi-card:hover { transform: none !important; box-shadow: none !important; }
    .stButton > button:hover { transform: none !important; box-shadow: none !important; }
    [data-testid="stSidebar"] .stButton > button:hover { transform: none !important; }
    .upload-zone:hover { background: rgba(30,136,229,0.05) !important; border-color: rgba(30,136,229,0.4) !important; }
}

/* ── Safe-area insets (iPhone notch / Dynamic Island) ── */
@supports (padding: max(0px)) {
    [data-testid="stMain"] {
        padding-left: max(0px, env(safe-area-inset-left)) !important;
        padding-right: max(0px, env(safe-area-inset-right)) !important;
        padding-bottom: max(0px, env(safe-area-inset-bottom)) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
for key, default in [
    ('authenticated', False), ('user_role', None), ('username', None),
    ('alert_thresholds', {'high_risk': 0.7, 'medium_risk': 0.5}),
    ('cnn_result', None), ('cnn_patient_id', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Manager init ───────────────────────────────────────────────────────────────
@st.cache_resource
def init_managers():
    return {
        'auth': AuthManager(),
        'db': DatabaseManager(),
        'models': AHFPredictionModels(),
        'notifications': NotificationManager(),
        'explainability': ExplainabilityManager(),
        'monitor': ModelMonitor(DatabaseManager()),
        'reports': ReportGenerator(DatabaseManager()),
        'validator': DataValidator(),
        'alerts': AlertSystem(DatabaseManager(), NotificationManager()),
        'data_generator': SyntheticDataGenerator(),
    }

@st.cache_resource
def get_cnn():
    return MedicalImageCNN()

managers = init_managers()

# ─── Helper: colored metric ─────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, color="#64b5f6"):
    delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def progress_bar(value, color="#1e88e5"):
    pct = int(value * 100)
    st.markdown(f"""
    <div class="prog-bar-wrap">
        <div class="prog-bar-fill" style="width:{pct}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def login_page():
    # Responsive: center column on all screens
    col_l, col_c, col_r = st.columns([0.1, 1, 0.1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; padding: 2rem 0 1.5rem 0;">
            <div style="font-size:4rem;">🫀</div>
            <div style="font-size:2rem; font-weight:800; color:#ffffff; margin-top:0.5rem;">CardioGuard AI</div>
            <div style="color:#78909c; font-size:1rem; margin-top:0.3rem;">Advanced AHF Rehospitalization Prediction Platform</div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑 Sign In", "📝 Register"])

        with tab_login:
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                col1, col2 = st.columns(2)
                with col1:
                    submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    user = managers['auth'].authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_role = user['role']
                        st.success(f"Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")

        with tab_reg:
            st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Min 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password")
                role = st.selectbox("Role", ["Doctor", "Nurse", "Admin"])
                email = st.text_input("Email Address", placeholder="you@hospital.com")
                submitted_reg = st.form_submit_button("Create Account", use_container_width=True)
                if submitted_reg:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters")
                    elif managers['auth'].create_user(new_username, new_password, role, email):
                        st.success("Account created! Please sign in.")
                    else:
                        st.error("Username already taken")

        st.markdown("""
        <div style="text-align:center; margin-top:2rem; color:#455a64; font-size:0.75rem;">
            Powered by CNN + XGBoost + Logistic Regression Ensemble
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 1rem 0 1.5rem 0; border-bottom: 1px solid rgba(30,136,229,0.2); margin-bottom: 1rem;">
            <div style="font-size:1.5rem; font-weight:800; color:#64b5f6;">🫀 CardioGuard</div>
            <div style="color:#90a4ae; font-size:0.8rem; margin-top:4px;">AI Clinical Decision Support</div>
            <div style="margin-top:0.8rem; padding:0.5rem 0.8rem; background:rgba(30,136,229,0.1); border-radius:8px; border:1px solid rgba(30,136,229,0.2);">
                <div style="color:#64b5f6; font-weight:600; font-size:0.9rem;">{st.session_state.username}</div>
                <div style="color:#78909c; font-size:0.75rem;">{st.session_state.user_role}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = [
            ("🏥", "Risk Assessment"),
            ("🔬", "Medical Imaging (CNN)"),
            ("📊", "Patient Monitoring"),
            ("📈", "Model Performance"),
            ("🚨", "Alerts & Notifications"),
            ("📄", "Reports"),
        ]
        if st.session_state.user_role == "Admin":
            pages.append(("⚙️", "System Administration"))

        selected = st.selectbox(
            "Navigation",
            [p[1] for p in pages],
            format_func=lambda x: next(f"{p[0]} {p[1]}" for p in pages if p[1] == x)
        )

        st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            for k in ['authenticated', 'user_role', 'username', 'cnn_result', 'cnn_patient_id']:
                st.session_state[k] = None if k != 'authenticated' else False
            st.rerun()

        # Quick stats
        try:
            db_stats = managers['db'].get_database_stats()
            st.markdown(f"""
            <div style="margin-top:1.5rem; padding:1rem; background:rgba(255,255,255,0.04); border-radius:12px; border:1px solid rgba(30,136,229,0.15);">
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.05em; color:#546e7a; margin-bottom:0.6rem;">Quick Stats</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                    <span style="color:#78909c; font-size:0.8rem;">Total Assessments</span>
                    <span style="color:#64b5f6; font-weight:600; font-size:0.8rem;">{db_stats.get('total_assessments', 0)}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                    <span style="color:#78909c; font-size:0.8rem;">Unique Patients</span>
                    <span style="color:#64b5f6; font-weight:600; font-size:0.8rem;">{db_stats.get('unique_patients', 0)}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#78909c; font-size:0.8rem;">Last 24h</span>
                    <span style="color:#64b5f6; font-weight:600; font-size:0.8rem;">{db_stats.get('assessments_24h', 0)}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# RISK ASSESSMENT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def risk_assessment_page():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏥 Patient Risk Assessment</div>
        <div class="hero-subtitle">30-Day AHF Rehospitalization Risk — Multi-Model Ensemble + CNN Imaging</div>
    </div>
    """, unsafe_allow_html=True)

    if not managers['models'].models_trained():
        with st.spinner("Training prediction models with synthetic data..."):
            train_models()

    # Show CNN scan result if available from imaging page
    if st.session_state.cnn_result and st.session_state.cnn_patient_id:
        cnn = st.session_state.cnn_result
        severity = cnn['severity_score']
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"""
            <div class="card" style="border-color:rgba(0,188,212,0.4); background:linear-gradient(145deg,rgba(0,188,212,0.08),rgba(0,188,212,0.03));">
                <div style="font-weight:700; color:#4dd0e1; margin-bottom:0.5rem;">🔬 Active CNN Scan — {cnn['scan_type']}</div>
                <div style="display:flex; gap:1rem; flex-wrap:wrap;">
                    <span class="finding-badge {'finding-normal' if 'No Significant' in cnn['primary_finding'] else 'finding-abnormal'}">
                        Primary: {cnn['primary_finding']}
                    </span>
                    {'<span class="finding-badge finding-warning">Secondary: ' + cnn['secondary_finding'] + '</span>' if cnn.get('secondary_finding') else ''}
                    <span class="finding-badge finding-warning">Severity: {severity:.0%}</span>
                    <span class="finding-badge finding-normal">Confidence: {cnn['confidence']:.0%}</span>
                </div>
                <div style="color:#78909c; font-size:0.8rem; margin-top:0.5rem;">
                    Imaging risk contribution: +{cnn['img_risk_contribution']:.1%} — will be included in ensemble score.
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            if st.button("🗑️ Clear Scan", use_container_width=True):
                st.session_state.cnn_result = None
                st.session_state.cnn_patient_id = None
                st.rerun()

    col_form, col_results = st.columns([1.4, 1])

    with col_form:
        section_header("👤", "Patient Demographics")
        with st.form("patient_assessment", clear_on_submit=False):
            c1, c2 = st.columns(2)
            with c1:
                patient_id = st.text_input("Patient ID", value=f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                age = st.number_input("Age (years)", 18, 100, 65)
                gender = st.selectbox("Gender", ["Male", "Female"])
                weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=0.5)
            with c2:
                nt_probnp = st.number_input("NT-proBNP (pg/mL)", 50.0, 50000.0, 2000.0, step=50.0)
                creatinine = st.number_input("Creatinine (mg/dL)", 0.5, 5.0, 1.2, step=0.1)
                ejection_fraction = st.number_input("Ejection Fraction (%)", 10, 80, 40)
                systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 120)

            section_header("🔊", "Ultrasound Parameters")
            u1, u2 = st.columns(2)
            with u1:
                b_line_score = st.number_input("B-line Score (0-28)", 0, 28, 8)
                ivc_collapsibility = st.number_input("IVC Collapsibility (%)", 0.0, 100.0, 50.0)
            with u2:
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 75)

            section_header("🩺", "Comorbidities")
            cc1, cc2 = st.columns(2)
            with cc1:
                diabetes = st.checkbox("Diabetes Mellitus")
                hypertension = st.checkbox("Hypertension")
            with cc2:
                ckd = st.checkbox("Chronic Kidney Disease")
                afib = st.checkbox("Atrial Fibrillation")

            submitted = st.form_submit_button("⚡ Assess Risk Now", use_container_width=True, type="primary")

            if submitted:
                patient_data = {
                    'patient_id': patient_id, 'age': age,
                    'gender': 1 if gender == "Male" else 0,
                    'weight': weight, 'nt_probnp': nt_probnp,
                    'creatinine': creatinine, 'b_line_score': b_line_score,
                    'ivc_collapsibility': ivc_collapsibility,
                    'ejection_fraction': ejection_fraction,
                    'systolic_bp': systolic_bp, 'heart_rate': heart_rate,
                    'diabetes': 1 if diabetes else 0,
                    'hypertension': 1 if hypertension else 0,
                    'ckd': 1 if ckd else 0, 'afib': 1 if afib else 0
                }

                validation_result = managers['validator'].validate_patient_data(patient_data)

                if validation_result['valid']:
                    with col_results:
                        with st.spinner("Running ensemble prediction..."):
                            predictions = managers['models'].predict_risk(patient_data)

                        lr_prob = predictions['logistic_regression']['probability']
                        xgb_prob = predictions['xgboost']['probability']
                        ensemble_prob = (lr_prob + xgb_prob) / 2

                        # Integrate CNN imaging if available
                        cnn_contribution = 0.0
                        if st.session_state.cnn_result:
                            cnn_contribution = st.session_state.cnn_result['img_risk_contribution']
                            # Weighted blend: 80% clinical + 20% imaging
                            ensemble_prob = min(1.0, ensemble_prob * 0.8 + cnn_contribution)

                        thresholds = st.session_state.alert_thresholds
                        if ensemble_prob >= thresholds['high_risk']:
                            risk_level, risk_color, risk_emoji = "High Risk", "#e53935", "🚨"
                            badge_cls = "risk-high"
                        elif ensemble_prob >= thresholds['medium_risk']:
                            risk_level, risk_color, risk_emoji = "Moderate Risk", "#fb8c00", "⚠️"
                            badge_cls = "risk-medium"
                        else:
                            risk_level, risk_color, risk_emoji = "Low Risk", "#43a047", "✅"
                            badge_cls = "risk-low"

                        # Save assessment
                        record = {**patient_data, 'gender': gender,
                                  'assessment_date': datetime.now().isoformat(),
                                  'lr_probability': lr_prob, 'xgb_probability': xgb_prob,
                                  'ensemble_probability': ensemble_prob, 'risk_level': risk_level}
                        managers['db'].save_assessment(record)
                        managers['alerts'].check_and_send_alerts(patient_data, ensemble_prob, risk_level)

                        display_risk_results(predictions, ensemble_prob, risk_level, risk_color,
                                             risk_emoji, badge_cls, patient_data, cnn_contribution)
                else:
                    st.error("Validation failed:")
                    for e in validation_result['errors']:
                        st.error(f"• {e}")

    with col_results:
        if not submitted:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1.5rem;">
                <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
                <div style="color:#546e7a; font-size:1rem;">Fill in the patient details and click <strong style="color:#64b5f6;">Assess Risk</strong> to see the prediction results here.</div>
                <div style="margin-top:1rem; color:#37474f; font-size:0.85rem;">Tip: Upload an X-ray first via the <strong style="color:#4dd0e1;">Medical Imaging</strong> page to enhance accuracy.</div>
            </div>
            """, unsafe_allow_html=True)


def display_risk_results(predictions, ensemble_prob, risk_level, risk_color,
                          risk_emoji, badge_cls, patient_data, cnn_contribution=0.0):
    section_header("📊", "Risk Assessment Results")

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ensemble_prob * 100, 1),
        number={'suffix': '%', 'font': {'size': 36, 'color': risk_color}},
        title={'text': "30-Day Readmission Risk", 'font': {'size': 13, 'color': '#90a4ae'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#546e7a', 'tickfont': {'color': '#546e7a'}},
            'bar': {'color': risk_color, 'thickness': 0.25},
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(255,255,255,0.05)',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(67,160,71,0.15)'},
                {'range': [50, 70], 'color': 'rgba(251,140,0,0.15)'},
                {'range': [70, 100], 'color': 'rgba(229,57,53,0.15)'}
            ],
            'threshold': {'line': {'color': '#ffffff', 'width': 3}, 'thickness': 0.8, 'value': ensemble_prob * 100}
        }
    ))
    fig_gauge.update_layout(
        height=240, margin=dict(t=30, b=10, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#90a4ae'}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown(f'<div style="text-align:center; margin:-0.5rem 0 1rem 0;"><span class="{badge_cls}">{risk_emoji} {risk_level}</span></div>', unsafe_allow_html=True)

    # Model breakdown
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Logistic Reg.", f"{predictions['logistic_regression']['probability']:.0%}", color="#64b5f6")
    with c2:
        kpi_card("XGBoost", f"{predictions['xgboost']['probability']:.0%}", color="#81c784")
    with c3:
        if cnn_contribution > 0:
            kpi_card("CNN Imaging", f"+{cnn_contribution:.0%}", color="#4dd0e1")
        else:
            kpi_card("Ensemble", f"{ensemble_prob:.0%}", color="#ffb74d")

    # SHAP explanation
    section_header("🔍", "Risk Factor Analysis")
    try:
        shap_exp = managers['explainability'].explain_prediction(patient_data, managers['models'])
        if shap_exp:
            shap_exp['plot'].update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#90a4ae'}, height=320,
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(shap_exp['plot'], use_container_width=True)
            top = shap_exp['top_factors']
            for factor, contrib in list(top.items())[:4]:
                icon = "▲" if contrib > 0 else "▼"
                col = "#ef9a9a" if contrib > 0 else "#81c784"
                st.markdown(f'<div style="font-size:0.8rem; color:{col}; margin:2px 0;">{icon} {factor}: {contrib:+.3f}</div>', unsafe_allow_html=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# MEDICAL IMAGING (CNN) PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def medical_imaging_page():
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1a237e 0%, #00695c 100%);">
        <div class="hero-title">🔬 Medical Imaging Analysis — CNN</div>
        <div class="hero-subtitle">Convolutional Neural Network · Chest X-Ray · ECG Reports · Scan Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    tab_upload, tab_history = st.tabs(["📤 Upload & Analyze", "📋 Scan History"])

    with tab_upload:
        col_upload, col_results = st.columns([1, 1.2])

        with col_upload:
            section_header("📤", "Upload Medical Image")

            st.markdown("""
            <div class="card">
                <div style="color:#90a4ae; font-size:0.85rem; margin-bottom:1rem;">
                    Upload a chest X-ray, CT scan, or other medical imaging report. The CNN will analyze it for cardiac and pulmonary findings relevant to AHF rehospitalization risk.
                </div>
                <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.5rem;">
                    <span class="finding-badge finding-normal">✓ Chest X-Ray</span>
                    <span class="finding-badge finding-normal">✓ CT Scan</span>
                    <span class="finding-badge finding-normal">✓ Echocardiogram</span>
                    <span class="finding-badge finding-warning">PNG / JPG / JPEG / WEBP</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            scan_type = st.selectbox(
                "Scan Type",
                ["Chest X-Ray", "CT Chest", "Echocardiogram", "Cardiac MRI", "Other Medical Report"],
                help="Select the type of medical image being uploaded"
            )

            patient_id_img = st.text_input(
                "Patient ID",
                value=st.session_state.cnn_patient_id or f"PAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Link this scan to a patient"
            )

            uploaded_file = st.file_uploader(
                "Drop your image here or browse",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                help="Medical imaging files — supports PNG, JPG, WEBP"
            )

            if uploaded_file:
                st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}",
                         use_container_width=True, clamp=True)

                analyze_btn = st.button("🧠 Run CNN Analysis", use_container_width=True, type="primary")

                if analyze_btn:
                    with st.spinner("Running CNN feature extraction and classification..."):
                        try:
                            cnn_model = get_cnn()
                            result = cnn_model.analyze(uploaded_file, scan_type=scan_type)

                            # Save to DB
                            managers['db'].save_image_scan({
                                'patient_id': patient_id_img,
                                'scan_date': datetime.now().isoformat(),
                                'scan_type': scan_type,
                                'primary_finding': result['primary_finding'],
                                'secondary_finding': result.get('secondary_finding'),
                                'confidence': result['confidence'],
                                'severity_score': result['severity_score'],
                                'img_risk_contribution': result['img_risk_contribution'],
                                'image_quality_label': result['image_quality']['label'],
                                'image_quality_score': result['image_quality']['score'],
                                'class_probabilities': result['class_probabilities'],
                                'recommendations': result['recommendations']
                            })

                            # Store in session for Risk Assessment page
                            st.session_state.cnn_result = result
                            st.session_state.cnn_patient_id = patient_id_img
                            st.rerun()

                        except Exception as e:
                            st.error(f"CNN analysis error: {str(e)}")

        with col_results:
            cnn = st.session_state.cnn_result
            if cnn:
                section_header("🧠", "CNN Analysis Results")

                # Primary finding banner
                primary = cnn['primary_finding']
                severity = cnn['severity_score']
                conf = cnn['confidence']
                is_normal = 'No Significant' in primary

                banner_color = "#1b5e20" if is_normal else ("#c62828" if severity > 0.7 else "#e65100")
                badge_txt = "NORMAL" if is_normal else ("HIGH SEVERITY" if severity > 0.7 else "MODERATE")

                st.markdown(f"""
                <div class="card" style="border-color:{'rgba(67,160,71,0.5)' if is_normal else 'rgba(229,57,53,0.5)'};
                     background:linear-gradient(145deg,{'rgba(67,160,71,0.1)' if is_normal else 'rgba(229,57,53,0.08)'},rgba(0,0,0,0.05));">
                    <div style="display:flex; justify-content:space-between; align-items:start;">
                        <div>
                            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:#78909c;">Primary Finding</div>
                            <div style="font-size:1.3rem; font-weight:700; color:#{'81c784' if is_normal else 'ef9a9a'}; margin-top:2px;">{primary}</div>
                            {'<div style="color:#ffcc80; font-size:0.85rem; margin-top:4px;">Secondary: ' + cnn['secondary_finding'] + '</div>' if cnn.get('secondary_finding') else ''}
                        </div>
                        <div style="background:{banner_color}; color:#fff; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:700; white-space:nowrap;">{badge_txt}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    kpi_card("Confidence", f"{conf:.0%}", color="#64b5f6")
                with m2:
                    kpi_card("Severity Score", f"{severity:.0%}", color="#ef9a9a" if severity > 0.6 else "#81c784")
                with m3:
                    kpi_card("Risk Contribution", f"+{cnn['img_risk_contribution']:.0%}", color="#ffb74d")

                # Finding probabilities
                section_header("📊", "CNN Classification Probabilities")
                probs = cnn['class_probabilities']
                for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                    bar_color = "#1e88e5" if label == primary else "#37474f"
                    st.markdown(f"""
                    <div style="margin:0.3rem 0;">
                        <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#90a4ae; margin-bottom:2px;">
                            <span>{"▶ " if label == primary else "&nbsp;&nbsp;"}{label}</span>
                            <span style="color:{'#64b5f6' if label == primary else '#546e7a'}; font-weight:{'700' if label == primary else '400'};">{prob:.1%}</span>
                        </div>
                        <div class="prog-bar-wrap">
                            <div class="prog-bar-fill" style="width:{prob*100:.1f}%; background:{'linear-gradient(90deg, #1e88e5, #0d47a1)' if label == primary else 'linear-gradient(90deg, #263238, #37474f)'};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Image quality
                quality = cnn['image_quality']
                q_color = {"Excellent": "#43a047", "Good": "#64b5f6",
                           "Acceptable": "#fb8c00", "Poor": "#e53935"}.get(quality['label'], "#78909c")
                st.markdown(f"""
                <div class="card" style="padding:0.8rem 1rem; margin-top:0.5rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#78909c; font-size:0.85rem;">Image Quality</span>
                        <span style="color:{q_color}; font-weight:700; font-size:0.85rem;">
                            <span class="status-dot" style="background:{q_color}; box-shadow:0 0 6px {q_color};"></span>
                            {quality['label']} ({quality['score']:.0%})
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Recommendations
                section_header("💊", "Clinical Recommendations")
                for rec in cnn['recommendations']:
                    is_urgent = 'URGENT' in rec
                    st.markdown(f'<div class="rec-item{"  urgent" if is_urgent else ""}">{"🚨 " if is_urgent else "• "}{rec}</div>', unsafe_allow_html=True)

                # Action buttons
                st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
                btn1, btn2 = st.columns(2)
                with btn1:
                    if st.button("📋 Use in Risk Assessment", use_container_width=True):
                        st.session_state.cnn_patient_id = patient_id_img
                        st.success("Scan linked! Go to Risk Assessment page.")
                with btn2:
                    if st.button("🗑️ Clear Results", use_container_width=True):
                        st.session_state.cnn_result = None
                        st.session_state.cnn_patient_id = None
                        st.rerun()

            else:
                st.markdown("""
                <div class="card" style="text-align:center; padding:4rem 2rem; min-height:400px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                    <div style="font-size:4rem; margin-bottom:1rem;">🧠</div>
                    <div style="color:#546e7a; font-size:1rem; margin-bottom:0.5rem;">Upload a medical image to run CNN analysis</div>
                    <div style="color:#37474f; font-size:0.85rem;">Supports chest X-rays, CT scans, echocardiograms and other medical imaging reports</div>
                    <div style="margin-top:2rem; padding:1rem; background:rgba(30,136,229,0.05); border-radius:12px; border:1px solid rgba(30,136,229,0.15); max-width:320px;">
                        <div style="color:#546e7a; font-size:0.8rem; font-weight:600; margin-bottom:0.5rem;">CNN ARCHITECTURE</div>
                        <div style="color:#37474f; font-size:0.78rem; line-height:1.6;">
                            Conv (8 filters) → ReLU → MaxPool<br>
                            Conv (16 filters) → ReLU → MaxPool<br>
                            Global Avg Pool → Flatten<br>
                            Logistic Classifier → 6 Classes
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_history:
        section_header("📋", "Previous Scans")
        scans = managers['db'].get_image_scans(limit=50)
        if scans:
            scan_df = pd.DataFrame(scans)
            cols_show = [c for c in ['scan_date', 'patient_id', 'scan_type', 'primary_finding',
                                      'severity_score', 'confidence', 'image_quality_label'] if c in scan_df.columns]
            disp_df = scan_df[cols_show].copy()
            if 'severity_score' in disp_df.columns:
                disp_df['severity_score'] = disp_df['severity_score'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            if 'confidence' in disp_df.columns:
                disp_df['confidence'] = disp_df['confidence'].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
            st.dataframe(disp_df, use_container_width=True)

            # Findings distribution
            if 'primary_finding' in scan_df.columns and len(scan_df) > 1:
                section_header("📊", "Findings Distribution")
                finding_counts = scan_df['primary_finding'].value_counts().reset_index()
                finding_counts.columns = ['Finding', 'Count']
                fig_pie = px.pie(finding_counts, values='Count', names='Finding',
                                  color_discrete_sequence=px.colors.sequential.Blues_r,
                                  hole=0.5)
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font={'color': '#90a4ae'}, height=300,
                                       margin=dict(t=20, b=20, l=20, r=20),
                                       showlegend=True)
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:2rem;">
                <div style="font-size:2rem;">🔬</div>
                <div style="color:#546e7a; margin-top:0.5rem;">No scans yet. Upload your first image to get started.</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PATIENT MONITORING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def patient_monitoring_page():
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1b2838 0%, #1a3a4a 100%);">
        <div class="hero-title">📊 Patient Monitoring Dashboard</div>
        <div class="hero-subtitle">Real-time tracking of risk scores, biomarkers, and trends</div>
    </div>
    """, unsafe_allow_html=True)

    assessments = managers['db'].get_all_assessments()
    if not assessments:
        st.markdown('<div class="card" style="text-align:center; padding:3rem;"><div style="font-size:2rem;">📭</div><div style="color:#546e7a; margin-top:0.5rem;">No assessments yet. Perform a risk assessment first.</div></div>', unsafe_allow_html=True)
        return

    df = pd.DataFrame(assessments)
    df['assessment_date'] = pd.to_datetime(df['assessment_date'])

    # Filters
    section_header("🔎", "Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        date_range = st.date_input("Date Range",
            value=(df['assessment_date'].min().date(), df['assessment_date'].max().date()))
    with f2:
        risk_filter = st.multiselect("Risk Level", ["Low Risk", "Moderate Risk", "High Risk"],
                                      default=["Low Risk", "Moderate Risk", "High Risk"])
    with f3:
        patient_filter = st.multiselect("Patient ID", df['patient_id'].unique(), default=[])

    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_df = df[(df['assessment_date'].dt.date >= date_range[0]) &
                          (df['assessment_date'].dt.date <= date_range[1]) &
                          (df['risk_level'].isin(risk_filter))]
    else:
        filtered_df = df[df['risk_level'].isin(risk_filter)]

    if patient_filter:
        filtered_df = filtered_df[filtered_df['patient_id'].isin(patient_filter)]

    # KPI row
    section_header("📈", "Summary Metrics")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Assessments", len(filtered_df), color="#64b5f6")
    with k2:
        hr_count = len(filtered_df[filtered_df['risk_level'] == 'High Risk'])
        kpi_card("High Risk Patients", hr_count, color="#ef9a9a")
    with k3:
        avg_risk = filtered_df['ensemble_probability'].mean() if len(filtered_df) > 0 else 0
        kpi_card("Avg Risk Score", f"{avg_risk:.1%}", color="#ffb74d")
    with k4:
        kpi_card("Unique Patients", filtered_df['patient_id'].nunique(), color="#81c784")

    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        section_header("📉", "Risk Score Distribution")
        fig_dist = px.histogram(filtered_df, x='ensemble_probability', nbins=20,
                                 color_discrete_sequence=["#1e88e5"],
                                 labels={'ensemble_probability': 'Risk Score', 'count': 'Patients'})
        fig_dist.add_vline(x=st.session_state.alert_thresholds['medium_risk'],
                           line_dash="dash", line_color="#fb8c00", annotation_text="Medium")
        fig_dist.add_vline(x=st.session_state.alert_thresholds['high_risk'],
                           line_dash="dash", line_color="#e53935", annotation_text="High")
        fig_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#90a4ae'}, height=280,
                                margin=dict(t=10, b=30, l=30, r=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    with ch2:
        section_header("🥧", "Risk Level Breakdown")
        risk_counts = filtered_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        color_map = {'High Risk': '#e53935', 'Moderate Risk': '#fb8c00', 'Low Risk': '#43a047'}
        fig_pie = px.pie(risk_counts, values='Count', names='Risk Level',
                          color='Risk Level', color_discrete_map=color_map, hole=0.5)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#90a4ae'},
                               height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Biomarker trends
    if len(filtered_df) > 1:
        section_header("📉", "Biomarker Trends")
        biomarker = st.selectbox("Select Biomarker",
            ["nt_probnp", "weight", "creatinine", "b_line_score", "ivc_collapsibility", "ejection_fraction"])
        fig_trend = px.scatter(filtered_df.sort_values('assessment_date'),
                               x='assessment_date', y=biomarker, color='risk_level',
                               size='ensemble_probability', hover_data=['patient_id'],
                               color_discrete_map={'Low Risk': '#43a047', 'Moderate Risk': '#fb8c00', 'High Risk': '#e53935'})
        fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font={'color': '#90a4ae'}, height=300,
                                  margin=dict(t=10, b=30, l=30, r=10))
        st.plotly_chart(fig_trend, use_container_width=True)

    # Table
    section_header("📋", "Recent Assessments")
    show_cols = [c for c in ['assessment_date', 'patient_id', 'risk_level', 'ensemble_probability',
                              'nt_probnp', 'weight', 'creatinine', 'age'] if c in filtered_df.columns]
    disp = filtered_df.nlargest(20, 'assessment_date')[show_cols].copy()
    if 'assessment_date' in disp.columns:
        disp['assessment_date'] = pd.to_datetime(disp['assessment_date']).dt.strftime('%Y-%m-%d %H:%M')
    if 'ensemble_probability' in disp.columns:
        disp['ensemble_probability'] = disp['ensemble_probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(disp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def model_performance_page():
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
        <div class="hero-title">📈 Model Performance Monitoring</div>
        <div class="hero-subtitle">CNN · XGBoost · Logistic Regression — Metrics, ROC Curves & Drift Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    if managers['models'].models_trained():
        metrics = managers['models'].get_performance_metrics()
        if metrics:
            section_header("🏆", "Model Performance Comparison")
            t1, t2, t3 = st.tabs(["Logistic Regression", "XGBoost", "CNN (Imaging)"])

            for tab, key, label in [(t1, 'logistic_regression', 'Logistic Regression'),
                                     (t2, 'xgboost', 'XGBoost')]:
                with tab:
                    m = metrics[key]
                    c1, c2, c3, c4 = st.columns(4)
                    for col, metric, val in [(c1, "Accuracy", f"{m['accuracy']:.3f}"),
                                              (c2, "AUC", f"{m['auc']:.3f}"),
                                              (c3, "Sensitivity", f"{m['sensitivity']:.3f}"),
                                              (c4, "Specificity", f"{m['specificity']:.3f}")]:
                        with col:
                            kpi_card(metric, val, color="#64b5f6")

            with t3:
                st.markdown("""
                <div class="card">
                    <div class="section-header">CNN Architecture Summary</div>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem;">
                        <div>
                            <div style="color:#78909c; font-size:0.8rem; margin-bottom:0.3rem;">Layer 1</div>
                            <div style="color:#90caf9;">Conv2D (8 filters, 3×3) → ReLU → MaxPool (2×2)</div>
                        </div>
                        <div>
                            <div style="color:#78909c; font-size:0.8rem; margin-bottom:0.3rem;">Layer 2</div>
                            <div style="color:#90caf9;">Conv2D (16 filters, 5×5) → ReLU → MaxPool (2×2)</div>
                        </div>
                        <div>
                            <div style="color:#78909c; font-size:0.8rem; margin-bottom:0.3rem;">Pooling</div>
                            <div style="color:#90caf9;">Global Average Pooling + Texture Features</div>
                        </div>
                        <div>
                            <div style="color:#78909c; font-size:0.8rem; margin-bottom:0.3rem;">Classifier</div>
                            <div style="color:#90caf9;">Multinomial Logistic Regression (6 classes)</div>
                        </div>
                    </div>
                    <div style="margin-top:1rem; padding:0.8rem; background:rgba(0,188,212,0.08); border-radius:10px; border:1px solid rgba(0,188,212,0.2);">
                        <div style="color:#4dd0e1; font-size:0.85rem; font-weight:600;">Output Classes</div>
                        <div style="color:#78909c; font-size:0.82rem; margin-top:0.4rem; line-height:1.8;">
                            No Significant Findings · Pulmonary Edema · Pleural Effusion<br>
                            Cardiomegaly · Vascular Congestion · Interstitial Infiltrates
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ROC Curves
            section_header("📉", "ROC Curves")
            roc_fig = managers['monitor'].create_roc_comparison(metrics)
            if roc_fig:
                roc_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font={'color': '#90a4ae'})
                st.plotly_chart(roc_fig, use_container_width=True)

            # Feature importance
            section_header("🎯", "Feature Importance (XGBoost)")
            importance = managers['models'].get_feature_importance()
            if importance:
                imp_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
                fig_imp = px.bar(imp_df.head(10), x='Importance', y='Feature', orientation='h',
                                  color='Importance', color_continuous_scale='Blues')
                fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                       font={'color': '#90a4ae'}, height=350,
                                       margin=dict(t=10, b=30, l=10, r=10))
                st.plotly_chart(fig_imp, use_container_width=True)

            # Confusion matrices
            section_header("🔢", "Confusion Matrices")
            cm1, cm2 = st.columns(2)
            for col, key, label in [(cm1, 'logistic_regression', 'Logistic Regression'),
                                     (cm2, 'xgboost', 'XGBoost')]:
                with col:
                    cm_fig = managers['monitor'].create_confusion_matrix_plot(
                        metrics[key]['confusion_matrix'], label)
                    if cm_fig:
                        cm_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                              font={'color': '#90a4ae'}, height=300)
                        st.plotly_chart(cm_fig, use_container_width=True)

        # Drift monitoring
        section_header("📡", "Model Drift Monitoring")
        drift = managers['monitor'].check_model_drift()
        if drift:
            drift['plot'].update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                         font={'color': '#90a4ae'})
            st.plotly_chart(drift['plot'], use_container_width=True)
        else:
            st.info("Insufficient data for drift analysis. Perform more assessments.")
    else:
        st.warning("Models not trained yet.")
        if st.button("Train Models Now"):
            train_models()


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS & NOTIFICATIONS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def alerts_notifications_page():
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #4a0000 0%, #7f0000 100%);">
        <div class="hero-title">🚨 Alerts & Notifications</div>
        <div class="hero-subtitle">Configure thresholds and manage clinical alert workflows</div>
    </div>
    """, unsafe_allow_html=True)

    section_header("⚙️", "Alert Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        high_risk_threshold = st.slider("High Risk Threshold", 0.5, 1.0,
                                         st.session_state.alert_thresholds['high_risk'], 0.05)
    with c2:
        medium_risk_threshold = st.slider("Medium Risk Threshold", 0.3, 0.8,
                                           st.session_state.alert_thresholds['medium_risk'], 0.05)
    if st.button("Save Thresholds"):
        st.session_state.alert_thresholds = {'high_risk': high_risk_threshold,
                                               'medium_risk': medium_risk_threshold}
        st.success("Thresholds updated!")

    section_header("📧", "Email Notifications")
    with st.form("notification_settings"):
        email_enabled = st.checkbox("Enable Email Notifications", value=True)
        notification_emails = st.text_area("Recipients (one per line)",
                                            value="doctor@hospital.com\nnurse@hospital.com")
        test_email = st.text_input("Test Email Address")
        s1, s2 = st.columns(2)
        with s1:
            save_s = st.form_submit_button("Save Settings")
        with s2:
            send_t = st.form_submit_button("Send Test")
        if save_s:
            emails = [e.strip() for e in notification_emails.split('\n') if e.strip()]
            managers['notifications'].update_notification_settings({'enabled': email_enabled, 'recipients': emails})
            st.success("Settings saved!")
        if send_t and test_email:
            ok = managers['notifications'].send_test_email(test_email)
            st.success(f"Test sent to {test_email}") if ok else st.error("Failed to send test email.")

    section_header("📋", "Recent Alerts")
    alerts = managers['alerts'].get_recent_alerts()
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.markdown('<div class="card" style="text-align:center; color:#546e7a; padding:1.5rem;">No recent alerts.</div>', unsafe_allow_html=True)

    section_header("📊", "Alert Statistics")
    stats = managers['alerts'].get_alert_statistics()
    if stats:
        a1, a2, a3 = st.columns(3)
        with a1: kpi_card("Total Alerts (24h)", stats.get('alerts_24h', 0))
        with a2: kpi_card("High Risk Alerts (7d)", stats.get('high_risk_7d', 0), color="#ef9a9a")
        with a3: kpi_card("Response Rate", f"{stats.get('response_rate', 0):.1%}", color="#81c784")


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def reports_page():
    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1c1c2e 0%, #2d2d44 100%);">
        <div class="hero-title">📄 Reports</div>
        <div class="hero-subtitle">Generate and download comprehensive clinical reports</div>
    </div>
    """, unsafe_allow_html=True)

    section_header("🖨️", "Generate Report")
    c1, c2 = st.columns(2)
    with c1:
        report_type = st.selectbox("Report Type",
            ["Daily Summary", "Weekly Summary", "Monthly Summary", "High Risk Patients", "Model Performance"])
    with c2:
        report_format = st.selectbox("Format", ["PDF", "CSV", "Excel"])

    if report_type in ["Weekly Summary", "Monthly Summary"]:
        d1, d2 = st.columns(2)
        with d1:
            date_from = st.date_input("From Date", value=datetime.now() - timedelta(days=7))
        with d2:
            date_to = st.date_input("To Date", value=datetime.now())
    else:
        date_from = date_to = datetime.now()

    if st.button("Generate Report", use_container_width=True):
        with st.spinner("Generating report..."):
            report_data = managers['reports'].generate_report(
                report_type.lower().replace(' ', '_'), date_from, date_to, report_format.lower())
            if report_data:
                st.success("Report generated!")
                if report_format == "PDF":
                    with open(report_data['filename'], 'rb') as f:
                        st.download_button("Download PDF", f.read(),
                                           file_name=report_data['filename'], mime="application/pdf")
                else:
                    st.download_button(f"Download {report_format}", report_data['data'],
                                       file_name=report_data['filename'], mime=report_data['mime_type'])
            else:
                st.error("Failed to generate report.")

    section_header("📁", "Recent Reports")
    recent = managers['reports'].get_recent_reports()
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True)
    else:
        st.markdown('<div class="card" style="text-align:center; color:#546e7a; padding:1.5rem;">No reports generated yet.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def admin_page():
    if st.session_state.user_role != "Admin":
        st.error("Access denied. Admin privileges required.")
        return

    st.markdown("""
    <div class="hero-banner" style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);">
        <div class="hero-title">⚙️ System Administration</div>
        <div class="hero-subtitle">Database management, user controls, and system settings</div>
    </div>
    """, unsafe_allow_html=True)

    section_header("🗄️", "Database Statistics")
    db_stats = managers['db'].get_database_stats()
    s1, s2, s3 = st.columns(3)
    with s1: kpi_card("Total Records", db_stats.get('total_assessments', 0))
    with s2: kpi_card("DB Size (MB)", f"{db_stats.get('db_size_mb', 0):.2f}", color="#ffb74d")
    with s3: kpi_card("Last Update", db_stats.get('last_assessment', 'N/A')[:10] if db_stats.get('last_assessment') else 'N/A', color="#81c784")

    section_header("👥", "User Management")
    users = managers['auth'].get_all_users()
    if users:
        users_df = pd.DataFrame(users)
        show_cols = [c for c in ['username', 'role', 'email', 'created_at'] if c in users_df.columns]
        st.dataframe(users_df[show_cols], use_container_width=True)

    section_header("🔧", "System Controls")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        if st.button("🔄 Retrain Models", use_container_width=True):
            train_models()
    with sc2:
        if st.button("📤 Export All Data", use_container_width=True):
            fn = managers['db'].export_to_csv()
            if fn:
                st.success(f"Exported to {fn}")
    with sc3:
        if st.button("💾 Backup Database", use_container_width=True):
            bp = managers['db'].backup_database()
            if bp:
                st.success(f"Backed up to {bp}")

    danger_zone = st.expander("⚠️ Danger Zone", expanded=False)
    with danger_zone:
        confirm = st.checkbox("I confirm I want to delete ALL patient data")
        if confirm:
            if st.button("🗑️ Clear All Records", use_container_width=True):
                managers['db'].clear_all_records()
                st.success("All records cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def train_models():
    with st.spinner("Generating synthetic data and training models..."):
        data = managers['data_generator'].generate_training_dataset(2000)
        managers['models'].train_models(data)
        metrics = managers['models'].get_performance_metrics()
        if metrics:
            managers['db'].save_model_performance('logistic_regression', metrics['logistic_regression'])
            managers['db'].save_model_performance('xgboost', metrics['xgboost'])
        st.success("Models trained successfully!")
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FLOW
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    login_page()
else:
    selected_page = render_sidebar()
    if selected_page == "Risk Assessment":
        risk_assessment_page()
    elif selected_page == "Medical Imaging (CNN)":
        medical_imaging_page()
    elif selected_page == "Patient Monitoring":
        patient_monitoring_page()
    elif selected_page == "Model Performance":
        model_performance_page()
    elif selected_page == "Alerts & Notifications":
        alerts_notifications_page()
    elif selected_page == "Reports":
        reports_page()
    elif selected_page == "System Administration":
        admin_page()
