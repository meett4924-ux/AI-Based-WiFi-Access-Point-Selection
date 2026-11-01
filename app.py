"""
app.py  -  AI Wi-Fi Optimizer
=============================
AI-Based Wi-Fi Access Point Selection Using Machine Learning
(Computer Networks Mini Project)

Run with:   streamlit run app.py

How the file is organised (top to bottom):
  1. Imports, constants and page configuration
  2. Custom CSS (white + rose-pink theme)
  3. Data helpers   (demo data, CSV validation)
  4. AI helpers     (scoring APs with the Random Forest, explanations)
  5. UI helpers     (header, KPI cards, AP cards, charts)
  6. Pages          (Overview, Access Points, AI Recommendation, ...)
  7. Main           (sidebar + navigation)
"""

import copy
import io
from datetime import datetime
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train_model import (
    CLASS_NAMES,
    CLASS_SCORES,
    FEATURES,
    LIMITS,
    SAMPLE_APS,
    ensure_model,
    normalized_components,
    quality_score,
)

# ==========================================================================
# 1. PAGE CONFIG + CONSTANTS
# ==========================================================================
st.set_page_config(
    page_title="AI Wi-Fi Optimizer",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colour palette from the project brief
ROSE, DARK_ROSE, LIGHT_ROSE = "#E75480", "#C83B68", "#FFF0F5"
TEXT, MUTED, BORDER = "#29232B", "#817681", "#F1DDE5"
SOFT_ROSE = "#F6BBCD"
SUCCESS, WARNING, ERROR = "#16A34A", "#F59E0B", "#EF4444"

# Colour for each predicted network-quality class
QUALITY_COLORS = {"Excellent": SUCCESS, "Good": "#65A30D", "Fair": WARNING, "Poor": ERROR}
AP_PALETTE = [ROSE, "#7A2E4E", "#F4A3BC", "#B8A1AE", "#C83B68", "#D9899F"]

SLIDER_LABELS = {
    "rssi": "RSSI (dBm)",
    "users": "Connected users",
    "bandwidth": "Bandwidth (Mbps)",
    "latency": "Latency (ms)",
    "congestion": "Congestion (%)",
}
DEFAULT_SIM = copy.deepcopy(SAMPLE_APS)  # starting values for the Simulation Lab


# 2. CUSTOM CSS
# ==========================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  --rose:#E75480; --dark:#C83B68; --light:#FFF0F5; --bg:#FFFDFE;
  --text:#29232B; --muted:#817681; --border:#F1DDE5;
  --ok:#16A34A; --warn:#F59E0B; --err:#EF4444;
  --shadow:0 4px 18px rgba(231,84,128,.08);
  --shadow-hover:0 14px 32px rgba(231,84,128,.18);
}

/* ---------- Page ---------- */
.stApp{
  background:
    radial-gradient(900px 420px at 88% -8%, #FFE4EC 0%, rgba(255,253,254,0) 65%),
    radial-gradient(700px 400px at -5% 105%, #FFEFF4 0%, rgba(255,253,254,0) 60%),
    #FFFDFE;
}
html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button,
[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"]{
  font-family:'Manrope',sans-serif;
}

/* ---------- Readability fix: force dark text on light surfaces ---------- */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.stMarkdown,
.stMarkdown p,
.stMarkdown li,
.stMarkdown span,
.stMarkdown strong,
.stMarkdown em,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stContainer"],
[data-testid="stExpander"]{
  color:var(--text);
}

/* Keep headings and regular Streamlit text readable inside cards/containers */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label,
[data-testid="stWidgetLabel"],
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"]{
  color:var(--text);
}

/* Streamlit inputs, select boxes, and data display */
.stApp input, .stApp textarea,
.stApp [data-baseweb="select"],
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricDelta"]{
  color:var(--text);
}

/* Avoid faded content caused by inherited opacity */
.stApp [data-testid="stVerticalBlock"],
.stApp [data-testid="stHorizontalBlock"],
.stApp [data-testid="stContainer"]{
  opacity:1;
}
h1,h2,h3,h4,h5{font-family:'Space Grotesk',sans-serif !important; color:var(--text); letter-spacing:-0.01em;}
.mono{font-family:'JetBrains Mono',monospace;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}
.block-container{padding-top:2.2rem; padding-bottom:3rem; max-width:1320px;}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:#FFF7FA;}
::-webkit-scrollbar-thumb{background:#F2BCCD;border-radius:10px;border:2px solid #FFF7FA;}
::-webkit-scrollbar-thumb:hover{background:var(--rose);}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#FFFFFF 0%,#FFF4F8 100%);
  border-right:1px solid var(--border);
}
.brand{display:flex;align-items:center;gap:12px;padding:.2rem .2rem 1.1rem;}
.brand-logo{width:44px;height:44px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,#E75480,#C83B68);box-shadow:0 8px 18px rgba(231,84,128,.35);}
.brand-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;color:var(--text);line-height:1.1;}
.brand-sub{font-size:.72rem;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-top:2px;}
.side-label{font-size:.68rem;font-weight:800;letter-spacing:.14em;color:var(--muted);margin:1.1rem 0 .35rem .2rem;}
a[data-testid="stPageLink-NavLink"]{
  border-radius:12px;padding:.5rem .8rem;margin-bottom:2px;border:1px solid transparent;transition:all .2s ease;
}
a[data-testid="stPageLink-NavLink"]:hover{background:var(--light);border-color:var(--border);transform:translateX(3px);}
a[data-testid="stPageLink-NavLink"][aria-current="page"]{
  background:linear-gradient(90deg,#FFE3EC,#FFF0F5);border-color:#F6C9D8;box-shadow:inset 3px 0 0 var(--rose);
}
a[data-testid="stPageLink-NavLink"] p{font-weight:600;color:var(--text);}
a[data-testid="stPageLink-NavLink"][aria-current="page"] p{color:var(--dark);font-weight:800;}
.side-footer{margin-top:1.6rem;padding:.9rem 1rem;border-radius:14px;background:#fff;border:1px solid var(--border);
  box-shadow:var(--shadow);font-size:.78rem;color:var(--muted);line-height:1.45;}
.side-footer b{color:var(--dark);display:block;font-size:.82rem;}

/* ---------- Header ---------- */
.page-eyebrow{font-size:.72rem;font-weight:800;letter-spacing:.16em;color:var(--rose);text-transform:uppercase;}
.hero-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2.35rem;line-height:1.1;color:var(--text);margin:.15rem 0 .25rem;}
.hero-title span{background:linear-gradient(90deg,#E75480,#C83B68);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero-sub{color:var(--muted);font-size:1rem;font-weight:500;}
.chip-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:.9rem;}
.chip{display:inline-flex;align-items:center;gap:6px;padding:.32rem .75rem;border-radius:999px;font-size:.76rem;font-weight:700;
  background:#fff;border:1px solid var(--border);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.03);}
.chip .dot{width:8px;height:8px;border-radius:50%;display:inline-block;}
.accent-line{height:4px;border-radius:4px;margin:1rem 0 1.5rem;background:linear-gradient(90deg,#E75480 0%,#F6BBCD 55%,rgba(246,187,205,0) 100%);}

/* ---------- Buttons ---------- */
.stButton>button, .stDownloadButton>button{
  width:100%;min-height:46px;border-radius:12px;font-weight:700;border:1px solid var(--border);background:#fff;color:var(--dark);
  transition:all .2s ease;box-shadow:0 1px 2px rgba(0,0,0,.04);
}
.stButton>button p, .stDownloadButton>button p{font-weight:700;}
.stButton>button:hover, .stDownloadButton>button:hover{
  transform:translateY(-2px);border-color:var(--rose);color:var(--dark);box-shadow:0 10px 22px rgba(231,84,128,.20);
}
.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,#E75480 0%,#C83B68 100%);color:#fff;border:none;box-shadow:0 8px 20px rgba(231,84,128,.35);
}
.stButton>button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p{color:#fff;}
.stButton>button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{
  color:#fff;box-shadow:0 14px 28px rgba(200,59,104,.42);filter:brightness(1.04);
}

/* ---------- Cards ---------- */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px;margin-bottom:1.4rem;}
.kpi-card{display:flex;gap:14px;align-items:center;background:#fff;border:1px solid var(--border);border-radius:18px;
  padding:1.05rem 1.15rem;box-shadow:var(--shadow);transition:all .25s ease;}
.kpi-card:hover{transform:translateY(-4px);box-shadow:var(--shadow-hover);border-color:#F2BCCD;}
.kpi-icon{flex:0 0 52px;height:52px;border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:1.5rem;
  background:linear-gradient(135deg,#FFE6EE,#FFF7FA);border:1px solid #F8D3E0;}
.kpi-label{font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.kpi-value{font-family:'Space Grotesk',sans-serif;font-size:1.65rem;font-weight:700;color:var(--text);line-height:1.15;}
.kpi-value small{font-size:.85rem;font-weight:600;color:var(--muted);margin-left:3px;}
.kpi-sub{font-size:.78rem;color:var(--muted);margin-top:1px;}

.ap-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px;margin-bottom:1.4rem;}
.ap-card{position:relative;background:#fff;border:1px solid var(--border);border-radius:20px;padding:1.2rem 1.25rem 1.1rem;
  box-shadow:var(--shadow);transition:all .25s ease;}
.ap-card:hover{transform:translateY(-4px);box-shadow:var(--shadow-hover);}
.ap-card.best{border:2px solid var(--rose);box-shadow:0 12px 34px rgba(231,84,128,.22);background:linear-gradient(180deg,#FFF8FA 0%,#FFFFFF 40%);}
.ap-ribbon{position:absolute;top:-12px;left:18px;background:linear-gradient(135deg,#E75480,#C83B68);color:#fff;font-size:.68rem;
  font-weight:800;letter-spacing:.08em;padding:.28rem .7rem;border-radius:999px;box-shadow:0 6px 14px rgba(231,84,128,.4);}
.ap-head{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.9rem;}
.ap-name{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.35rem;color:var(--text);}
.ap-ssid{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--muted);}
.badge{display:inline-flex;align-items:center;gap:6px;padding:.25rem .7rem;border-radius:999px;font-size:.74rem;font-weight:800;}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:.9rem;}
.metric{background:#FFF9FB;border:1px solid #F8E6EC;border-radius:12px;padding:.55rem .7rem;}
.metric-label{font-size:.66rem;font-weight:800;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.metric-value{font-family:'JetBrains Mono',monospace;font-size:.98rem;font-weight:600;color:var(--text);}
.score-row{display:flex;justify-content:space-between;align-items:baseline;font-size:.8rem;font-weight:700;color:var(--muted);margin-bottom:5px;}
.score-row b{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;color:var(--dark);}
.bar{height:8px;border-radius:8px;background:#F8E6EC;overflow:hidden;}
.bar-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#F6BBCD,#E75480);}

.reco-card{position:relative;overflow:hidden;border-radius:22px;padding:1.5rem 1.7rem;margin-bottom:.8rem;color:var(--text);
  background:linear-gradient(135deg,#FFE3EC 0%,#FFF6F9 55%,#FFFFFF 100%);border:1.5px solid #F2BCCD;box-shadow:0 14px 36px rgba(231,84,128,.16);}
.reco-card:after{content:"";position:absolute;right:-60px;top:-60px;width:220px;height:220px;border-radius:50%;
  background:radial-gradient(circle,rgba(231,84,128,.18),rgba(231,84,128,0) 70%);}
.reco-eyebrow{font-size:.74rem;font-weight:800;letter-spacing:.16em;color:var(--dark);}
.reco-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:2.1rem;margin:.3rem 0 .1rem;}
.reco-sub{color:var(--muted);font-weight:600;}
.reco-stats{display:flex;flex-wrap:wrap;gap:12px;margin-top:1rem;position:relative;z-index:1;}
.reco-stat{background:rgba(255,255,255,.85);border:1px solid #F6D3DF;border-radius:14px;padding:.55rem .95rem;}
.reco-stat span{display:block;font-size:.66rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
.reco-stat b{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;color:var(--text);}

.reason{display:flex;gap:12px;align-items:flex-start;background:#fff;border:1px solid var(--border);border-radius:14px;
  padding:.75rem .95rem;margin-bottom:9px;transition:all .2s ease;}
.reason:hover{border-color:#F2BCCD;transform:translateX(3px);}
.reason .ico{flex:0 0 30px;height:30px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-weight:800;}
.reason .ico.good{background:#DCFCE7;color:var(--ok);} .reason .ico.warn{background:#FEF3C7;color:#B45309;}
.reason-title{font-weight:800;color:var(--text);} .reason-detail{font-size:.85rem;color:var(--muted);}

.banner{border-radius:14px;padding:.8rem 1.1rem;font-size:.9rem;font-weight:600;margin-bottom:1rem;border:1px solid;}
.banner.info{background:#FFF7FA;border-color:#F6C9D8;color:var(--dark);}
.banner.warn{background:#FFFBEB;border-color:#FDE68A;color:#92400E;}
.section-title{font-family:'Space Grotesk',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text);margin:1.2rem 0 .2rem;}
.section-sub{color:var(--muted);font-size:.9rem;margin-bottom:.8rem;}

/* keyed containers act as white cards (charts, simulation panels) */
div[class*="st-key-card"]{background:#fff;border:1px solid var(--border);border-radius:20px;padding:1rem 1.2rem;
  box-shadow:var(--shadow);transition:box-shadow .25s ease,border-color .25s ease;margin-bottom:.6rem;}
div[class*="st-key-card"]:hover{box-shadow:var(--shadow-hover);border-color:#F2BCCD;}

/* ---------- Architecture diagram ---------- */
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:.6rem 0 1rem;}
.flow-node{flex:1 1 130px;min-width:130px;text-align:center;background:#fff;border:1px solid var(--border);border-radius:16px;
  padding:.9rem .6rem;box-shadow:var(--shadow);font-weight:800;color:var(--text);transition:all .2s ease;}
.flow-node:hover{transform:translateY(-3px);border-color:var(--rose);}
.flow-node .em{font-size:1.5rem;display:block;margin-bottom:3px;}
.flow-node small{display:block;font-weight:500;color:var(--muted);font-size:.74rem;margin-top:2px;}
.flow-arrow{font-size:1.5rem;font-weight:800;color:var(--rose);}
.tech-chip{display:inline-block;margin:4px 6px 4px 0;padding:.4rem .85rem;border-radius:999px;background:var(--light);
  border:1px solid #F6C9D8;color:var(--dark);font-weight:700;font-size:.82rem;}

/* ---------- Widgets ---------- */
div[data-testid="stSlider"] [role="slider"]{background:var(--rose);box-shadow:0 0 0 4px rgba(231,84,128,.18);}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--border);border-radius:16px;padding:.8rem 1rem;box-shadow:var(--shadow);}
div[data-testid="stExpander"]{border-radius:14px;border-color:var(--border);}

/* ---------- Fix dark Streamlit toast/status bar after Refresh ---------- */
/* Toasts are normally dark in Streamlit; this forces readable light styling. */
[data-testid="stToast"],
[data-testid="stToast"] *,
div[role="alert"][data-testid],
div[role="alert"] [data-testid="stMarkdownContainer"] {
  background-color: #FFF0F5 !important;
  color: #29232B !important;
  border-color: #F1DDE5 !important;
  opacity: 1 !important;
}

/* Streamlit toast content can include nested paragraphs and SVG icons. */
[data-testid="stToast"] p,
[data-testid="stToast"] span,
[data-testid="stToast"] div,
[data-testid="stToast"] a {
  color: #29232B !important;
}

/* Keep standard alert/status content legible without dark text on dark fill. */
[data-testid="stAlert"] {
  background: #FFF7FA !important;
  border: 1px solid #F6C9D8 !important;
}
[data-testid="stAlert"] *,
[data-testid="stStatusWidget"] *,
[data-testid="stNotification"] * {
  color: #29232B !important;
  opacity: 1 !important;
}

</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def h(markup):
    """Remove indentation so Streamlit's Markdown does not treat HTML as a code block."""
    return "".join(line.strip() for line in markup.splitlines())


# ==========================================================================
# 3. DATA HELPERS
# ==========================================================================
@st.cache_resource(show_spinner="Loading AI model (training it first if needed)...")
def get_bundle():
    """Load the saved Random Forest (or train it automatically on first run)."""
    return ensure_model()


def init_state():
    """Create all session-state values the first time the app starts."""
    ss = st.session_state
    ss.setdefault("applied", copy.deepcopy(DEFAULT_SIM))  # values used by the dashboard
    ss.setdefault("data_mode", "Demo Data")
    ss.setdefault("show_details", False)
    ss.setdefault("sim_ran", False)
    ss.setdefault("csv_df", None)
    ss.setdefault("csv_errors", [])
    ss.setdefault("last_refresh", datetime.now())


def sim_to_df(applied):
    """Turn the {'AP1': {...}} dictionary into a DataFrame."""
    rows = []
    for ap, vals in applied.items():
        rows.append({"AP": ap, "SSID": f"WIFI_{ap}", **vals})
    return pd.DataFrame(rows)


def sample_csv_bytes():
    sample = pd.DataFrame([{"AP": ap, **vals} for ap, vals in SAMPLE_APS.items()])
    return sample.to_csv(index=False).encode("utf-8")


def validate_csv(uploaded_file):
    """
    Read and validate an uploaded CSV.
    Returns (DataFrame, []) when valid or (None, [list of error messages]).
    """
    try:
        raw = uploaded_file.getvalue()
        if not raw.strip():
            return None, ["The uploaded file is empty."]
        df = pd.read_csv(io.BytesIO(raw))
    except pd.errors.EmptyDataError:
        return None, ["The uploaded file is empty."]
    except Exception as exc:  # unreadable / not a CSV
        return None, [f"Could not read the file as a CSV: {exc}"]

    if df.empty:
        return None, ["The CSV has column headers but no data rows."]

    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["ap"] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, [
            f"Missing required column(s): {', '.join(missing)}.",
            "Expected columns: AP, rssi, users, bandwidth, latency, congestion",
        ]
    df = df[required].copy()

    errors = []
    for col in FEATURES:  # data types
        numeric = pd.to_numeric(df[col], errors="coerce")
        bad_rows = (numeric.isna()).to_numpy().nonzero()[0] + 2  # +2 = header + 1-based rows
        if len(bad_rows):
            shown = ", ".join(str(r) for r in bad_rows[:5])
            errors.append(f"Column '{col}' has non-numeric or blank values (CSV row {shown}).")
        df[col] = numeric
    if errors:
        return None, errors

    allowed = {  # sensible physical ranges
        "rssi": (-100, -20), "users": (0, 500), "bandwidth": (0, 1000),
        "latency": (0, 2000), "congestion": (0, 100),
    }
    for col, (low, high) in allowed.items():  # invalid values
        out = df[(df[col] < low) | (df[col] > high)]
        if not out.empty:
            errors.append(f"Column '{col}' must be between {low} and {high} (found {out[col].iloc[0]} in CSV row {out.index[0] + 2}).")

    df["ap"] = df["ap"].astype(str).str.strip()
    if (df["ap"] == "").any() or (df["ap"].str.lower() == "nan").any():
        errors.append("The 'AP' column contains blank names.")
    if df["ap"].duplicated().any():
        errors.append("The 'AP' column contains duplicate names. Each AP must appear once.")
    if len(df) < 2:
        errors.append("Provide at least 2 access points so the AI has something to compare.")
    if len(df) > 12:
        errors.append("A maximum of 12 access points is supported.")
    if errors:
        return None, errors

    df = df.rename(columns={"ap": "AP"})
    df.insert(1, "SSID", "WIFI_" + df["AP"])
    return df.reset_index(drop=True), []


def get_active_data():
    """Return (DataFrame, source_label, source_kind) for the selected data mode."""
    ss = st.session_state
    if ss.data_mode == "CSV Upload" and ss.csv_df is not None:
        return ss.csv_df, "CSV Upload — offline data, not live traffic", "csv"
    if ss.data_mode == "CSV Upload":
        return sim_to_df(ss.applied), "Demo Mode — no valid CSV loaded yet", "fallback"
    return sim_to_df(ss.applied), "Demo Mode — Simulated Network Data", "demo"


def refresh_data():
    """Refresh button: generate slightly different SIMULATED measurements."""
    ss = st.session_state
    ss.last_refresh = datetime.now()
    rng = np.random.default_rng()
    spread = {"rssi": 3, "users": 2, "bandwidth": 5, "latency": 8, "congestion": 6}
    new_values = {}
    for ap, vals in ss.applied.items():
        new_values[ap] = {}
        for feat in FEATURES:
            low, high = LIMITS[feat]
            moved = vals[feat] + rng.normal(0, spread[feat])
            new_values[ap][feat] = int(np.clip(round(moved), low, high))
            ss[f"sim_{ap}_{feat}"] = new_values[ap][feat]  # keep sliders in sync
    ss.applied = new_values
    ss.toast_msg = "Simulated measurements refreshed"


def run_optimization():
    """Simulation Lab -> 'Run AI Optimization' button callback."""
    ss = st.session_state
    ss.applied = {ap: {f: int(ss[f"sim_{ap}_{f}"]) for f in FEATURES} for ap in DEFAULT_SIM}
    ss.sim_ran = True
    ss.last_refresh = datetime.now()


def reset_simulation():
    """Simulation Lab -> 'Reset Simulation' button callback."""
    ss = st.session_state
    for ap, vals in DEFAULT_SIM.items():
        for feat, value in vals.items():
            ss[f"sim_{ap}_{feat}"] = value
    ss.applied = copy.deepcopy(DEFAULT_SIM)
    ss.sim_ran = False


def toggle_details():
    st.session_state.show_details = not st.session_state.show_details


# ==========================================================================
# 4. AI HELPERS
# ==========================================================================
def score_aps(df, bundle):
    """
    Run the Random Forest on every AP.

    The classifier outputs a probability for each quality class
    (Poor / Fair / Good / Excellent). We turn that into one AI score
    (0-100) = sum( probability_of_class x class_value ).
    The AP with the highest AI score is recommended. A tiny fraction of the
    rule-based quality score is added only to break exact ties.
    """
    model = bundle["model"]
    probabilities = model.predict_proba(df[FEATURES].astype(float))
    classes = [int(c) for c in model.classes_]
    class_values = np.array([CLASS_SCORES[c] for c in classes])

    out = df.copy().reset_index(drop=True)
    out["ai_score"] = probabilities @ class_values
    out["quality"] = [CLASS_NAMES[classes[i]] for i in probabilities.argmax(axis=1)]
    out["confidence"] = probabilities.max(axis=1) * 100
    out["rule_score"] = quality_score(out).to_numpy()
    for i, c in enumerate(classes):
        out[f"p_{CLASS_NAMES[c]}"] = probabilities[:, i] * 100

    rank_value = out["ai_score"] + 0.01 * out["rule_score"]  # tie-breaker only
    out["is_best"] = False
    out.loc[rank_value.idxmax(), "is_best"] = True
    return out


def explain_recommendation(scored):
    """Compare the recommended AP with the other APs and return reasons."""
    best = scored[scored["is_best"]].iloc[0]
    others = scored[~scored["is_best"]]
    norm = normalized_components(scored)
    best_norm = norm[scored["is_best"]].iloc[0]
    other_norm = norm[~scored["is_best"]].mean()

    wording = {
        "rssi": ("Stronger signal", "{b:.0f} dBm compared with {o:.0f} dBm average of the other APs"),
        "congestion": ("Lower congestion", "{b:.0f}% compared with {o:.0f}% average of the other APs"),
        "bandwidth": ("More available bandwidth", "{b:.0f} Mbps compared with {o:.0f} Mbps average of the other APs"),
        "latency": ("Lower latency", "{b:.0f} ms compared with {o:.0f} ms average of the other APs"),
        "users": ("Fewer connected users", "{b:.0f} users compared with {o:.0f} average on the other APs"),
    }
    wording_bad = {
        "rssi": "Weaker signal", "congestion": "Higher congestion", "bandwidth": "Less bandwidth",
        "latency": "Higher latency", "users": "More connected users",
    }

    reasons, tradeoffs = [], []
    for feat in FEATURES:
        diff = best_norm[feat] - other_norm[feat]
        detail = wording[feat][1].format(b=best[feat], o=others[feat].mean())
        if diff > 0.03:
            reasons.append((diff, wording[feat][0], detail))
        elif diff < -0.05:
            tradeoffs.append((diff, wording_bad[feat], detail))
    reasons.sort(key=lambda r: -r[0])
    tradeoffs.sort(key=lambda r: r[0])

    ranked = scored.sort_values("ai_score", ascending=False).reset_index(drop=True)
    margin = None
    if len(ranked) > 1:
        margin = (ranked.loc[0, "ai_score"] - ranked.loc[1, "ai_score"], ranked.loc[1, "AP"])
    return reasons, tradeoffs, margin


# ==========================================================================
# 5. UI HELPERS
# ==========================================================================
def render_header(page_name, scored, source_label, source_kind):
    """Title, subtitle, status chips and Refresh button."""
    ss = st.session_state
    if ss.pop("toast_msg", None):
        st.toast("Simulated measurements refreshed", icon="🔄")

    avg_score = scored["ai_score"].mean()
    if avg_score >= 65:
        status, dot = "Network Healthy", SUCCESS
    elif avg_score >= 45:
        status, dot = "Network Moderate", WARNING
    else:
        status, dot = "Network Degraded", ERROR
    source_dot = ROSE if source_kind == "demo" else (WARNING if source_kind == "fallback" else "#2563EB")
    now = datetime.now().strftime("%d %b %Y, %I:%M %p")

    left, right = st.columns([5, 1.3], vertical_alignment="center")
    with left:
        st.markdown(
            h(f"""
            <div class="page-eyebrow">{escape(page_name)}</div>
            <div class="hero-title">AI <span>Wi-Fi</span> Optimizer</div>
            <div class="hero-sub">Intelligent Access Point Selection &amp; Network Performance Analytics</div>
            <div class="chip-row">
              <span class="chip">🕒 {now}</span>
              <span class="chip"><span class="dot" style="background:{dot}"></span>{status}</span>
              <span class="chip"><span class="dot" style="background:{source_dot}"></span>{escape(source_label)}</span>
            </div>"""),
            unsafe_allow_html=True,
        )
    with right:
        st.button("Refresh Data", icon=":material/refresh:", key=f"refresh_{page_name}", on_click=refresh_data,
                  help="Generate fresh simulated measurements (Demo mode)")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)


def kpi_html(icon, label, value, sub, unit=""):
    unit_html = f"<small>{unit}</small>" if unit else ""
    return h(f"""
        <div class="kpi-card"><div class="kpi-icon">{icon}</div><div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}{unit_html}</div>
        <div class="kpi-sub">{sub}</div></div></div>""")


def render_kpis(scored):
    best = scored[scored["is_best"]].iloc[0]
    cards = [
        kpi_html("📡", "Total Access Points", len(scored), "Monitored in current dataset"),
        kpi_html("👥", "Connected Devices", int(scored["users"].sum()), "Users across all APs"),
        kpi_html("🏆", "Best Access Point", escape(str(best["AP"])), f"AI score {best['ai_score']:.1f} / 100"),
        kpi_html("📶", "Avg Signal Strength", f"{scored['rssi'].mean():.0f}", "Mean RSSI across APs", "dBm"),
        kpi_html("⏱️", "Avg Latency", f"{scored['latency'].mean():.0f}", "Mean response time", "ms"),
        kpi_html("🚀", "Available Bandwidth", f"{scored['bandwidth'].sum():.0f}", "Combined across all APs", "Mbps"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def quality_badge(quality):
    color = QUALITY_COLORS[quality]
    return f'<span class="badge" style="color:{color};background:{color}1A;border:1px solid {color}40">● {quality}</span>'


def render_ap_cards(scored):
    cards = []
    for _, ap in scored.iterrows():
        best_class = " best" if ap["is_best"] else ""
        ribbon = '<div class="ap-ribbon">⭐ AI RECOMMENDED</div>' if ap["is_best"] else ""
        width = float(np.clip(ap["ai_score"], 0, 100))
        metrics = [
            ("📶 Signal", f"{ap['rssi']:.0f} dBm"), ("👥 Users", f"{ap['users']:.0f}"),
            ("🚀 Bandwidth", f"{ap['bandwidth']:.0f} Mbps"), ("⏱️ Latency", f"{ap['latency']:.0f} ms"),
            ("🚦 Congestion", f"{ap['congestion']:.0f}%"), ("🧠 Confidence", f"{ap['confidence']:.0f}%"),
        ]
        metric_html = "".join(
            f'<div class="metric"><div class="metric-label">{lbl}</div><div class="metric-value">{val}</div></div>'
            for lbl, val in metrics
        )
        cards.append(h(f"""
            <div class="ap-card{best_class}">{ribbon}
              <div class="ap-head"><div><div class="ap-name">{escape(str(ap['AP']))}</div>
              <div class="ap-ssid">SSID: {escape(str(ap['SSID']))}</div></div>{quality_badge(ap['quality'])}</div>
              <div class="metric-grid">{metric_html}</div>
              <div class="score-row"><span>AI SCORE</span><b>{ap['ai_score']:.1f}</b></div>
              <div class="bar"><div class="bar-fill" style="width:{width}%"></div></div>
            </div>"""))
    st.markdown(f'<div class="ap-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_reco_card(scored, key):
    """Recommendation card + 'View Recommendation Details' button."""
    best = scored[scored["is_best"]].iloc[0]
    st.markdown(
        h(f"""
        <div class="reco-card">
          <div class="reco-eyebrow">🏆 AI RECOMMENDED ACCESS POINT</div>
          <div class="reco-title">{escape(str(best['AP']))} — {escape(str(best['SSID']))}</div>
          <div class="reco-sub">Best predicted network quality</div>
          <div class="reco-stats">
            <div class="reco-stat"><span>AI Score</span><b>{best['ai_score']:.1f} / 100</b></div>
            <div class="reco-stat"><span>Predicted Quality</span><b style="color:{QUALITY_COLORS[best['quality']]}">{best['quality']}</b></div>
            <div class="reco-stat"><span>Model Confidence</span><b>{best['confidence']:.0f}%</b></div>
          </div></div>"""),
        unsafe_allow_html=True,
    )
    label = "Hide Recommendation Details" if st.session_state.show_details else "View Recommendation Details"
    left, _ = st.columns([1.2, 3])
    with left:
        st.button(label, key=key, on_click=toggle_details, icon=":material/psychology:")
    if st.session_state.show_details:
        render_reasons(scored)


def render_reasons(scored):
    reasons, tradeoffs, margin = explain_recommendation(scored)
    best = scored[scored["is_best"]].iloc[0]
    st.markdown(f"#### Why {best['AP']} was recommended")
    if margin:
        st.caption(f"The model scored {best['AP']} {margin[0]:.1f} points above the runner-up ({margin[1]}).")
    if not reasons:
        reasons = [(0, "Balanced overall profile", "No single metric dominates, but the combined pattern scores highest.")]
    html_parts = [
        f'<div class="reason"><div class="ico good">✓</div><div><div class="reason-title">{t}</div><div class="reason-detail">{d}</div></div></div>'
        for _, t, d in reasons
    ]
    html_parts += [
        f'<div class="reason"><div class="ico warn">!</div><div><div class="reason-title">Trade-off: {t}</div><div class="reason-detail">{d}</div></div></div>'
        for _, t, d in tradeoffs
    ]
    st.markdown(h("".join(html_parts)), unsafe_allow_html=True)


def banner(text, kind="info"):
    st.markdown(f'<div class="banner {kind}">{text}</div>', unsafe_allow_html=True)


# ---------- Charts ----------
def style_fig(fig, title, height=340, legend=False):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=17, color=TEXT), x=0.01),
        height=height, margin=dict(l=15, r=15, t=58, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=TEXT), showlegend=legend,
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(family="Manrope", color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#F8E6EC", zeroline=False, tickfont=dict(color=MUTED))
    return fig


def bar_colors(scored):
    return [ROSE if best else SOFT_ROSE for best in scored["is_best"]]


def bar_chart(scored, column, title, unit, tip):
    values = scored[column].astype(float)
    if column == "rssi":
        # RSSI is negative: draw bars rising from -100 dBm so a taller bar = stronger signal
        fig = go.Figure(go.Bar(
            x=scored["AP"], y=values + 100, base=-100, marker_color=bar_colors(scored),
            customdata=values, text=[f"{v:.0f} {unit}" for v in values], textposition="outside",
            hovertemplate="<b>%{x}</b><br>RSSI: %{customdata:.0f} dBm<extra></extra>",
        ))
        fig.update_yaxes(range=[-100, -20], title="RSSI (dBm)")
    else:
        fig = go.Figure(go.Bar(
            x=scored["AP"], y=values, marker_color=bar_colors(scored),
            text=[f"{v:.0f} {unit}" for v in values], textposition="outside",
            hovertemplate="<b>%{x}</b><br>" + title.split(" ")[0] + ": %{y:.0f} " + unit + "<extra></extra>",
        ))
        fig.update_yaxes(range=[0, max(values.max() * 1.25, 1)], title=unit)
    fig.update_traces(textfont=dict(family="JetBrains Mono", size=12, color=TEXT), cliponaxis=False)
    style_fig(fig, title)
    fig.add_annotation(text=tip, xref="paper", yref="paper", x=0.01, y=-0.18, showarrow=False,
                       font=dict(size=11, color=MUTED), xanchor="left")
    return fig


def radar_chart(scored):
    norm = normalized_components(scored) * 100
    axes = ["Signal", "Few Users", "Bandwidth", "Low Latency", "Low Congestion"]
    fig = go.Figure()
    for i, (_, ap) in enumerate(scored.iterrows()):
        values = norm.loc[ap.name, FEATURES].tolist()
        color = AP_PALETTE[i % len(AP_PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=values + values[:1], theta=axes + axes[:1], name=str(ap["AP"]) + (" ⭐" if ap["is_best"] else ""),
            fill="toself", opacity=0.55 if ap["is_best"] else 0.35, line=dict(color=color, width=3 if ap["is_best"] else 2),
            hovertemplate="%{theta}: %{r:.0f}/100<extra>" + str(ap["AP"]) + "</extra>",
        ))
    style_fig(fig, "Network Quality Radar (normalized, 100 = best)", height=380, legend=True)
    fig.update_layout(
        polar=dict(bgcolor="rgba(255,240,245,.5)",
                   radialaxis=dict(range=[0, 100], gridcolor="#F2D5DF", tickfont=dict(size=9, color=MUTED)),
                   angularaxis=dict(gridcolor="#F2D5DF", tickfont=dict(size=12, color=TEXT))),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
    )
    return fig


def ai_score_chart(scored):
    values = scored["ai_score"]
    fig = go.Figure(go.Bar(
        x=scored["AP"], y=values, marker_color=bar_colors(scored),
        text=[f"{v:.1f}" for v in values], textposition="outside",
        customdata=scored[["quality", "confidence"]].to_numpy(),
        hovertemplate="<b>%{x}</b><br>AI score: %{y:.1f}<br>Quality: %{customdata[0]}<br>Confidence: %{customdata[1]:.0f}%<extra></extra>",
    ))
    fig.update_traces(textfont=dict(family="JetBrains Mono", size=13, color=TEXT), cliponaxis=False)
    style_fig(fig, "AI Score Comparison (recommended AP highlighted)")
    fig.update_yaxes(range=[0, 108], title="AI score (0-100)")
    best = scored[scored["is_best"]].iloc[0]
    fig.add_annotation(x=best["AP"], y=min(best["ai_score"] + 12, 105), text="★ Recommended", showarrow=False,
                       font=dict(color=DARK_ROSE, size=12, family="Manrope"))
    return fig


def probability_chart(scored):
    fig = go.Figure()
    for name in CLASS_NAMES:
        col = f"p_{name}"
        if col in scored:
            fig.add_trace(go.Bar(
                y=scored["AP"], x=scored[col], name=name, orientation="h", marker_color=QUALITY_COLORS[name],
                hovertemplate="<b>%{y}</b><br>" + name + ": %{x:.0f}%<extra></extra>",
            ))
    style_fig(fig, "Model Probability per Quality Class", height=300, legend=True)
    fig.update_layout(barmode="stack", legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(range=[0, 100], title="Probability (%)")
    return fig


def confusion_chart(metrics):
    z, labels = metrics["confusion_matrix"], metrics["labels"]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=z, texttemplate="%{text}", showscale=False,
        colorscale=[[0, "#FFF7FA"], [1, DARK_ROSE]], textfont=dict(family="JetBrains Mono", size=14),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Samples: %{z}<extra></extra>",
    ))
    style_fig(fig, "Confusion Matrix (test set)", height=380)
    fig.update_xaxes(title="Predicted class", side="bottom")
    fig.update_yaxes(title="Actual class", autorange="reversed")
    return fig


def importance_chart(metrics):
    names = {"rssi": "RSSI", "users": "Users", "bandwidth": "Bandwidth", "latency": "Latency", "congestion": "Congestion"}
    items = sorted(metrics["feature_importance"].items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v * 100 for _, v in items], y=[names[k] for k, _ in items], orientation="h",
        marker_color=ROSE, text=[f"{v*100:.1f}%" for _, v in items], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_traces(textfont=dict(family="JetBrains Mono", size=12), cliponaxis=False)
    style_fig(fig, "Feature Importance (learned by the model)", height=380)
    fig.update_xaxes(range=[0, max(v for _, v in items) * 100 * 1.25], title="Importance (%)")
    return fig


def show_chart(fig, key):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})


def chart_card(fig, key):
    with st.container(key=f"card_{key}"):
        show_chart(fig, key)


# ==========================================================================
# 6. PAGES
# ==========================================================================
def load_context():
    """Score the active dataset and return everything a page needs."""
    df, label, kind = get_active_data()
    scored = score_aps(df, get_bundle())
    return scored, label, kind


def page_overview():
    scored, label, kind = load_context()
    render_header("Overview", scored, label, kind)
    render_kpis(scored)
    render_reco_card(scored, key="details_overview")
    st.markdown('<div class="section-title">Performance at a glance</div>'
                '<div class="section-sub">AI score and normalized quality profile of every access point.</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chart_card(ai_score_chart(scored), "overview_score")
    with c2:
        chart_card(radar_chart(scored), "overview_radar")


def page_access_points():
    scored, label, kind = load_context()
    render_header("Access Points", scored, label, kind)
    banner("Each card is generated from the active dataset. The AI-recommended AP has a rose border.")
    render_ap_cards(scored)

    st.markdown('<div class="section-title">AP comparison table</div>', unsafe_allow_html=True)
    table = scored[["AP", "SSID", "rssi", "users", "bandwidth", "latency", "congestion", "quality", "ai_score", "is_best"]].copy()
    table["is_best"] = table["is_best"].map({True: "⭐ Recommended", False: ""})
    table.columns = ["AP", "SSID", "RSSI (dBm)", "Users", "Bandwidth (Mbps)", "Latency (ms)", "Congestion (%)",
                     "Quality", "AI Score", "Result"]
    st.dataframe(table, hide_index=True, column_config={
        "AI Score": st.column_config.ProgressColumn("AI Score", min_value=0, max_value=100, format="%.1f"),
    })

    results = scored[["AP", "SSID", *FEATURES, "quality", "confidence", "ai_score", "is_best"]].copy()
    results = results.rename(columns={"quality": "predicted_quality", "confidence": "confidence_pct", "is_best": "recommended"})
    results = results.round(2)
    c1, c2, _ = st.columns([1.3, 1.3, 2.4])
    with c1:
        st.download_button("Download AP Comparison", results.to_csv(index=False).encode("utf-8"),
                           "ap_comparison_results.csv", "text/csv", key="dl_results", icon=":material/download:")
    with c2:
        st.download_button("Download Sample Dataset", sample_csv_bytes(), "sample_ap_input.csv", "text/csv",
                           key="dl_sample_page", icon=":material/description:")


def page_ai_recommendation():
    scored, label, kind = load_context()
    bundle = get_bundle()
    metrics = bundle["metrics"]
    render_header("AI Recommendation", scored, label, kind)
    render_reco_card(scored, key="details_ai")

    st.markdown('<div class="section-title">How the AI decides</div>', unsafe_allow_html=True)
    st.markdown(
        "1. Each AP's five measurements (RSSI, users, bandwidth, latency, congestion) are sent to the **Random Forest**.  \n"
        "2. The model returns a probability for every quality class: *Poor, Fair, Good, Excellent*.  \n"
        "3. Probabilities are converted into one **AI score (0-100)** so APs can be compared fairly.  \n"
        "4. The AP with the highest AI score is recommended."
    )
    chart_card(probability_chart(scored), "ai_probabilities")

    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)
    banner(f"⚠️ This model was trained on <b>synthetic (simulated)</b> Wi-Fi data — {metrics['n_train']:,} training and "
           f"{metrics['n_test']:,} test samples (trained {bundle['trained_at']}). All metrics below are computed from real "
           "predictions on the held-out test set.", "warn")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("Precision (weighted)", f"{metrics['precision'] * 100:.1f}%")
    m3.metric("Recall (weighted)", f"{metrics['recall'] * 100:.1f}%")
    m4.metric("F1-score (weighted)", f"{metrics['f1'] * 100:.1f}%")
    c1, c2 = st.columns(2)
    with c1:
        chart_card(confusion_chart(metrics), "ai_confusion")
    with c2:
        chart_card(importance_chart(metrics), "ai_importance")


def page_analytics():
    scored, label, kind = load_context()
    render_header("Network Analytics", scored, label, kind)
    banner("Interactive charts — hover for details. The recommended AP is shown in the strongest rose colour.")
    row1 = st.columns(2)
    with row1[0]:
        chart_card(bar_chart(scored, "rssi", "Signal Strength Comparison", "dBm", "Taller bar = stronger signal (closer to 0 dBm)"), "an_rssi")
    with row1[1]:
        chart_card(bar_chart(scored, "users", "Network Load (Connected Users)", "users", "Fewer users = lighter load"), "an_users")
    row2 = st.columns(2)
    with row2[0]:
        chart_card(bar_chart(scored, "bandwidth", "Available Bandwidth", "Mbps", "Higher is better"), "an_bw")
    with row2[1]:
        chart_card(bar_chart(scored, "latency", "Latency Comparison", "ms", "Lower is better"), "an_lat")
    row3 = st.columns(2)
    with row3[0]:
        chart_card(radar_chart(scored), "an_radar")
    with row3[1]:
        chart_card(ai_score_chart(scored), "an_score")


def page_simulation_lab():
    ss = st.session_state
    sim_scored = score_aps(sim_to_df(ss.applied), get_bundle())
    render_header("Simulation Lab", sim_scored, "Demo Mode — Simulated Network Data", "demo")
    banner("🧪 <b>Demo Mode — Simulated Network Data.</b> Adjust each AP, then press <b>Run AI Optimization</b>. "
           "Nothing here is read from a live network.")
    if ss.data_mode == "CSV Upload" and ss.csv_df is not None:
        banner("The other pages currently show your uploaded CSV. Switch the sidebar to <b>Demo Data</b> to see these "
               "simulation values there.", "warn")

    # Slider keys are removed by Streamlit when this page is left, so re-create them from the applied values.
    for ap, vals in ss.applied.items():
        for feat in FEATURES:
            ss.setdefault(f"sim_{ap}_{feat}", vals[feat])

    columns = st.columns(len(ss.applied))
    for col, ap in zip(columns, ss.applied):
        with col, st.container(key=f"card_sim_{ap}"):
            st.markdown(f'<div class="ap-name">{ap}</div><div class="ap-ssid">SSID: WIFI_{ap}</div>', unsafe_allow_html=True)
            for feat in FEATURES:
                low, high = LIMITS[feat]
                st.slider(SLIDER_LABELS[feat], min_value=low, max_value=high, step=1, key=f"sim_{ap}_{feat}")

    pending = any(int(ss[f"sim_{ap}_{f}"]) != ss.applied[ap][f] for ap in ss.applied for f in FEATURES)
    st.write("")
    b1, b2, spacer = st.columns([1.4, 1.2, 3])
    with b1:
        st.button("Run AI Optimization", type="primary", key="btn_run", on_click=run_optimization, icon=":material/bolt:")
    with b2:
        st.button("Reset Simulation", key="btn_reset", on_click=reset_simulation, icon=":material/restart_alt:")
    if pending:
        banner("Slider values changed — press <b>Run AI Optimization</b> to update the recommendation.", "warn")
    elif ss.sim_ran:
        banner("✅ Optimization complete — the AI re-evaluated all access points with your values.")

    # Results always reflect the last applied (optimized) values
    sim_scored = score_aps(sim_to_df(ss.applied), get_bundle())
    st.markdown('<div class="section-title">Optimization results</div>', unsafe_allow_html=True)
    render_kpis(sim_scored)
    render_reco_card(sim_scored, key="details_sim")
    c1, c2 = st.columns(2)
    with c1:
        chart_card(ai_score_chart(sim_scored), "sim_score")
    with c2:
        chart_card(radar_chart(sim_scored), "sim_radar")
    if not ss.show_details:
        render_reasons(sim_scored)


def page_about():
    scored, label, kind = load_context()
    render_header("About Project", scored, label, kind)
    st.markdown("## AI-Based Wi-Fi Access Point Selection Using Machine Learning")
    st.markdown("A **Computer Networks mini project** that recommends the best Wi-Fi Access Point using network "
                "performance parameters: signal strength (RSSI), connected users, available bandwidth, latency and congestion.")

    st.markdown('<div class="section-title">Network architecture (Cisco Packet Tracer)</div>', unsafe_allow_html=True)
    st.markdown(h("""
        <div class="flow">
          <div class="flow-node"><span class="em">📱</span>Wireless Clients<small>6 devices</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">📡</span>Access Points<small>AP1 · AP2 · AP3</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">🔀</span>Switch<small>1 switch</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">🌐</span>Router<small>1 router</small></div>
        </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="section-title">AI workflow (Python + Streamlit)</div>', unsafe_allow_html=True)
    st.markdown(h("""
        <div class="flow">
          <div class="flow-node"><span class="em">📊</span>Network Measurements<small>Demo / CSV</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">⚙️</span>Data Processing<small>Validate &amp; normalize</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">🌲</span>Random Forest Model<small>Quality prediction</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">🏆</span>AP Recommendation<small>Highest AI score</small></div><div class="flow-arrow">→</div>
          <div class="flow-node"><span class="em">🖥️</span>Dashboard<small>Charts &amp; reasons</small></div>
        </div>"""), unsafe_allow_html=True)

    banner("Cisco Packet Tracer is used to simulate and demonstrate the wireless network topology. Python and Streamlit "
           "implement the AI-based AP selection logic. The current prototype uses simulated network measurements or CSV "
           "input and does not automatically control Packet Tracer client roaming.")

    c1, c2 = st.columns(2)
    with c1, st.container(key="card_about_topology"):
        st.markdown("#### Packet Tracer topology")
        st.markdown("- 1 Router\n- 1 Switch\n- 3 Wireless Access Points (WIFI_AP1, WIFI_AP2, WIFI_AP3)\n- 6 Wireless client devices\n\n"
                    "Used for network visualization and connectivity demonstration.")
    with c2, st.container(key="card_about_ai"):
        st.markdown("#### AI model")
        st.markdown("- **Random Forest Classifier** (scikit-learn, 200 trees)\n- Inputs: RSSI, users, bandwidth, latency, congestion\n"
                    "- Output: Poor / Fair / Good / Excellent probabilities → AI score\n- Trained on **synthetic data**; saved with Joblib")

    st.markdown('<div class="section-title">Technology stack</div>', unsafe_allow_html=True)
    techs = ["Python", "Streamlit", "Pandas", "NumPy", "Scikit-learn", "Joblib", "Plotly", "Custom CSS", "Google Fonts", "Cisco Packet Tracer"]
    st.markdown("".join(f'<span class="tech-chip">{t}</span>' for t in techs), unsafe_allow_html=True)


# ==========================================================================
# 7. MAIN: sidebar + navigation
# ==========================================================================
WIFI_SVG = ('<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="#fff" stroke-width="2.2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M2 8.8a15 15 0 0 1 20 0"/>'
            '<path d="M5 12.5a10 10 0 0 1 14 0"/><path d="M8.5 16a5 5 0 0 1 7 0"/>'
            '<circle cx="12" cy="19.5" r="1.2" fill="#fff"/></svg>')


def render_data_source_controls():
    """Sidebar: choose Demo data or upload a CSV."""
    ss = st.session_state
    st.markdown('<div class="side-label">DATA SOURCE</div>', unsafe_allow_html=True)
    st.radio("Data source", ["Demo Data", "CSV Upload"], key="data_mode", label_visibility="collapsed")
    if ss.data_mode == "CSV Upload":
        uploaded = st.file_uploader("Upload AP data (CSV)", type=["csv"], key="csv_file",
                                    help="Columns: AP, rssi, users, bandwidth, latency, congestion")
        st.download_button("Download sample CSV", sample_csv_bytes(), "sample_ap_input.csv", "text/csv",
                           key="dl_sample_sidebar", icon=":material/download:")
        if uploaded is not None:
            ss.csv_df, ss.csv_errors = validate_csv(uploaded)
        else:
            ss.csv_df, ss.csv_errors = None, []
        for message in ss.csv_errors:
            st.error(message, icon="🚫")
        if ss.csv_df is not None:
            st.success(f"Loaded {len(ss.csv_df)} access points from CSV.", icon="✅")
            st.caption("Offline file — not live network traffic.")
        elif uploaded is None:
            st.caption("No file yet — showing demo data until a valid CSV is uploaded.")
    else:
        st.caption("Simulated measurements. Edit them in the Simulation Lab.")


def main():
    inject_css()
    init_state()
    get_bundle()  # trains the model on the very first run if needed

    pages = [
        st.Page(page_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(page_access_points, title="Access Points", icon=":material/router:", url_path="access-points"),
        st.Page(page_ai_recommendation, title="AI Recommendation", icon=":material/psychology:", url_path="ai-recommendation"),
        st.Page(page_analytics, title="Network Analytics", icon=":material/insights:", url_path="network-analytics"),
        st.Page(page_simulation_lab, title="Simulation Lab", icon=":material/science:", url_path="simulation-lab"),
        st.Page(page_about, title="About Project", icon=":material/info:", url_path="about-project"),
    ]
    navigation = st.navigation(pages, position="hidden")  # we draw our own branded menu below

    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="brand-logo">{WIFI_SVG}</div><div><div class="brand-name">AI Wi-Fi Optimizer</div>'
            '<div class="brand-sub">Network Intelligence</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">NAVIGATION</div>', unsafe_allow_html=True)
        for page in pages:
            st.page_link(page)
        render_data_source_controls()
        st.markdown(
            '<div class="side-footer"><b>AI-Based Wi-Fi AP Selection</b>Computer Networks Mini Project</div>',
            unsafe_allow_html=True,
        )

    try:
        navigation.run()
    except Exception as exc:  # show a friendly message instead of a crash
        st.error(f"Something went wrong while building this page: {exc}")
        st.exception(exc)


main()
