import streamlit as st
import numpy as np
import json
import pickle
import tensorflow
import os
import time
import io
from PIL import Image
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CitrusMD · Disease Detection",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme toggle state ───────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode

# ── CSS Variables based on theme ────────────────────────────────────
if dark:
    css_vars = """
    :root {
        --bg:        #0a0f0a;
        --bg2:       #0f160f;
        --bg3:       #141e14;
        --surface:   #182018;
        --surface2:  #1c261c;
        --border:    #223022;
        --border2:   #2e4a2e;
        --accent:    #4ecb5e;
        --accent2:   #3aad4a;
        --accent3:   #a3ebb0;
        --gold:      #e8b84b;
        --red:       #f05252;
        --orange:    #f08040;
        --purple:    #9b6aef;
        --teal:      #3dc0a8;
        --text:      #edf2ed;
        --text2:     #8aaa8a;
        --text3:     #405840;
        --text4:     #2a3e2a;
        --shadow:    rgba(0,0,0,0.5);
        --card-hover: #1e2a1e;
    }
    """
else:
    css_vars = """
    :root {
        --bg:        #f4f7f4;
        --bg2:       #edf2ed;
        --bg3:       #e4ece4;
        --surface:   #ffffff;
        --surface2:  #f8fbf8;
        --border:    #d0ddd0;
        --border2:   #b4ccb4;
        --accent:    #2a8a38;
        --accent2:   #1e7a2c;
        --accent3:   #1a6028;
        --gold:      #b07820;
        --red:       #c43030;
        --orange:    #c05820;
        --purple:    #6a3ec0;
        --teal:      #1a8878;
        --text:      #1a2a1a;
        --text2:     #3a5a3a;
        --text3:     #6a8a6a;
        --text4:     #9aaa9a;
        --shadow:    rgba(0,0,0,0.12);
        --card-hover: #f0f5f0;
    }
    """

st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
    font-size: 15px;
}

.main { background: var(--bg) !important; }
.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1280px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Nav radio ── */
[data-testid="stSidebar"] .stRadio > label { display: none; }
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 1.2rem !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: var(--text2) !important;
    cursor: pointer;
    transition: all 0.15s;
    margin: 0 0.5rem;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: var(--surface2) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio input:checked + label {
    background: var(--surface) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    border: 1px solid var(--border2) !important;
}

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
    box-shadow: 0 2px 12px rgba(78,203,94,0.2) !important;
}
.stButton > button:hover {
    background: var(--accent2) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(78,203,94,0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { color: var(--text2) !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1.5px solid var(--border2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-size: 0.88rem !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--accent) !important;
    background: var(--surface2) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

/* ── Metrics ── */
div[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Outfit', sans-serif !important; }
div[data-testid="stMetricLabel"] { color: var(--text2) !important; }

/* ── Progress ── */
[data-testid="stProgressBar"] > div { background: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 10px; }

/* ── Components ── */
.page-header {
    margin-bottom: 2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.page-header-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.page-header-title {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.03em;
    line-height: 1.1;
}
.page-header-title span { color: var(--accent); }
.page-header-sub {
    font-size: 0.9rem;
    color: var(--text2);
    margin-top: 0.4rem;
    line-height: 1.5;
}

.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text3);
    margin: 1.5rem 0 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s, background 0.2s;
}
.card:hover { border-color: var(--border2); background: var(--card-hover); }

.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.stat-lbl {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text3);
    margin-top: 0.15rem;
}

.consensus-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-bottom: 1rem;
    border-top: 3px solid var(--accent);
}
.cons-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}
.cons-name {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    margin-bottom: 0.25rem;
}
.cons-cause {
    font-size: 0.83rem;
    color: var(--text2);
    line-height: 1.5;
}
.cons-agree {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    margin-top: 0.75rem;
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 100px;
    padding: 0.2rem 0.85rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent);
}

.pred-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem 0.9rem 1.2rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.15s, background 0.15s;
}
.pred-card:hover { border-color: var(--border2); background: var(--card-hover); }
.pred-accent-bar {
    width: 3px;
    height: 36px;
    border-radius: 3px;
    flex-shrink: 0;
}
.pred-model-name {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text3);
}
.pred-disease-name {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text);
    margin: 0.1rem 0;
}
.pred-conf-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.75rem;
    color: var(--text2);
    margin-top: 0.1rem;
}
.conf-track {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
}

.rec-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
}
.rec-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 0.75rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}
.rec-item {
    font-size: 0.85rem;
    color: var(--text2);
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    line-height: 1.55;
    display: flex;
    gap: 0.5rem;
}
.rec-item:last-child { border-bottom: none; }
.rec-bullet {
    color: var(--accent);
    font-weight: 700;
    flex-shrink: 0;
}

.chip {
    display: inline-flex;
    align-items: center;
    border-radius: 100px;
    padding: 0.22rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.chip-high   { background: rgba(240,82,82,0.1);  border: 1px solid rgba(240,82,82,0.3);  color: var(--red); }
.chip-medium { background: rgba(240,128,64,0.1); border: 1px solid rgba(240,128,64,0.3); color: var(--orange); }
.chip-low    { background: rgba(78,203,94,0.1);  border: 1px solid rgba(78,203,94,0.3);  color: var(--accent); }

.warn-card {
    background: var(--surface);
    border: 1px solid rgba(240,82,82,0.3);
    border-left: 3px solid var(--red);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.warn-title { font-size: 1rem; font-weight: 700; color: var(--red); margin-bottom: 0.4rem; }
.warn-sub   { font-size: 0.85rem; color: var(--text2); line-height: 1.55; }

.perf-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.perf-tile:hover { transform: translateY(-2px); border-color: var(--border2); }
.perf-model { font-size: 0.8rem; font-weight: 600; color: var(--text2); margin-bottom: 0.4rem; }
.perf-pct { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.perf-dec { font-size: 0.75rem; color: var(--text3); margin-top: 0.25rem; font-variant-numeric: tabular-nums; }

.about-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
}
.about-title { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.15rem; }
.about-sub   { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text3); margin-bottom: 0.75rem; }
.about-desc  { font-size: 0.86rem; color: var(--text2); line-height: 1.6; margin-bottom: 0.9rem; }
.about-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.about-col-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.about-item  { font-size: 0.82rem; color: var(--text2); padding: 0.18rem 0; }

.disease-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.8rem;
}
.disease-name  { font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.1rem; }
.disease-cause { font-size: 0.78rem; color: var(--text3); margin-bottom: 0.7rem; }
.symptom-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.8rem;
    font-size: 0.83rem;
    color: var(--text2);
}
.symp-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text3); margin-bottom: 0.2rem; }

/* ── Upload placeholder ── */
.upload-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 280px;
    background: var(--surface);
    border: 2px dashed var(--border2);
    border-radius: 14px;
    gap: 0.6rem;
}
.upload-placeholder-icon { font-size: 2.5rem; opacity: 0.4; }
.upload-placeholder-text { font-size: 0.9rem; color: var(--text3); font-weight: 500; }

/* ── Setup card ── */
.setup-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.setup-num {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: var(--bg3);
    border: 1px solid var(--border2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    color: var(--accent);
    font-size: 0.85rem;
    flex-shrink: 0;
}
.setup-title { font-weight: 700; color: var(--text); font-size: 0.92rem; margin-bottom: 0.15rem; }
.setup-desc  { font-size: 0.82rem; color: var(--text2); }

/* ── Sidebar logo ── */
.sb-logo {
    padding: 1.4rem 1.2rem 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.6rem;
}
.sb-logo-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
}
.sb-logo-title span { color: var(--accent); }
.sb-logo-sub {
    font-size: 0.7rem;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.1rem;
}
.sb-section {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text4);
    padding: 0.5rem 1.2rem 0.3rem;
}
.sb-acc-row {
    padding: 0.5rem 1.2rem;
}
.sb-acc-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 0.2rem;
}
.sb-acc-name { color: var(--text2); font-weight: 500; }
.sb-acc-val  { font-weight: 700; font-variant-numeric: tabular-nums; }
.sb-acc-bar {
    height: 3px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}
.sb-acc-fill {
    height: 100%;
    border-radius: 3px;
}
.sb-footer {
    padding: 1rem 1.2rem;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}
.sb-footer-text {
    font-size: 0.7rem;
    color: var(--text4);
    line-height: 1.7;
}

/* ── Theme toggle ── */
.theme-btn-wrap .stButton > button {
    background: var(--surface) !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-size: 0.8rem !important;
    padding: 0.45rem 1rem !important;
    width: auto !important;
    border-radius: 8px !important;
}
.theme-btn-wrap .stButton > button:hover {
    border-color: var(--border2) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .page-header-title { font-size: 1.5rem; }
    .about-grid { grid-template-columns: 1fr; }
    .perf-pct { font-size: 1.8rem; }
    .cons-name { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────────────
import os
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CLASSES      = ["Canker", "Greening", "Black_Spot", "Nutrient_Deficiency"]
CLASS_LABELS = ["Canker", "Greening", "Black Spot", "Nutrient Deficiency"]
IMG_SIZE     = (224, 224)

DISEASE_INFO = {
    "Canker": {
        "display": "Citrus Canker",
        "severity": "high",
        "cause": "Xanthomonas citri subsp. citri — bacterial",
        "symptoms": "Raised corky lesions with yellow halos on leaves, stems and fruit",
        "color": "#f05252",
        "recommendations": [
            "Apply copper-based bactericides (copper hydroxide or copper oxychloride) every 14 days",
            "Prune and destroy infected branches — sterilise tools with 70% alcohol between cuts",
            "Switch to drip irrigation to reduce leaf wetness and bacterial spread",
            "Quarantine infected trees; do not move plant material to other areas",
            "Control citrus leafminer — it creates entry wounds for bacteria",
            "Report to local agricultural extension — Canker is a notifiable disease",
        ],
    },
    "Greening": {
        "display": "Citrus Greening (HLB)",
        "severity": "high",
        "cause": "Candidatus Liberibacter asiaticus — spread by Asian citrus psyllid",
        "symptoms": "Asymmetric blotchy mottling, vein yellowing, small misshapen bitter fruit",
        "color": "#f08040",
        "recommendations": [
            "No cure exists — infected trees must be removed and destroyed immediately",
            "Apply systemic insecticides to control Asian citrus psyllid (Diaphorina citri)",
            "Replace removed trees with certified disease-free nursery stock",
            "Monitor neighbouring trees weekly for early psyllid infestation",
            "Use reflective mulches to deter psyllids from landing on young trees",
            "Alert your regional plant health authority immediately",
        ],
    },
    "Black_Spot": {
        "display": "Citrus Black Spot",
        "severity": "medium",
        "cause": "Phyllosticta citricarpa — fungal",
        "symptoms": "Dark necrotic spots with yellow halo on fruit rind; sunken lesions on leaves",
        "color": "#9b6aef",
        "recommendations": [
            "Apply protectant fungicides (mancozeb, carbendazim) from petal fall through fruit maturity",
            "Improve canopy aeration through targeted pruning to reduce humidity",
            "Remove and destroy fallen leaves and infected fruit — spores persist in leaf litter",
            "Apply post-harvest thiabendazole or imazalil wax coating before storage or export",
            "Maintain a spray calendar — apply every 3–4 weeks during wet season",
            "Note: Black Spot is a quarantine pest — affected fruit may face export restrictions",
        ],
    },
    "Nutrient_Deficiency": {
        "display": "Nutrient Deficiency",
        "severity": "low",
        "cause": "Deficiency in Zinc, Nitrogen, Magnesium, Iron or other micronutrients",
        "symptoms": "Interveinal chlorosis, yellowing, pale or stunted new growth",
        "color": "#3dc0a8",
        "recommendations": [
            "Conduct soil and leaf tissue analysis to identify the specific deficient nutrient",
            "Apply balanced NPK foliar spray (20-20-20) for general deficiency",
            "Zinc deficiency: apply zinc sulphate foliar spray (0.3%) every 3 weeks",
            "Magnesium deficiency: apply Epsom salt (magnesium sulphate) 2% foliar spray",
            "Iron chlorosis: apply chelated iron (EDTA-Fe) to soil or as foliar application",
            "Check soil pH — most nutrients unavailable above pH 7.5; apply sulfur to acidify",
        ],
    },
}


# ── Model loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    required = [
        f"{MODELS_DIR}/cnn_model.h5",
        f"{MODELS_DIR}/svm_model.pkl",
        f"{MODELS_DIR}/rf_model.pkl",
        f"{MODELS_DIR}/knn_model.pkl",
        f"{MODELS_DIR}/scaler.pkl",
        f"{MODELS_DIR}/metadata.json",  
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, None, None, f"Missing files: {', '.join(missing)}"

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        cnn    = load_model(f"{MODELS_DIR}/cnn_model.h5")
        svm    = pickle.load(open(f"{MODELS_DIR}/svm_model.pkl",  "rb"))
        rf     = pickle.load(open(f"{MODELS_DIR}/rf_model.pkl",   "rb"))
        knn    = pickle.load(open(f"{MODELS_DIR}/knn_model.pkl",  "rb"))
        scaler = pickle.load(open(f"{MODELS_DIR}/scaler.pkl",     "rb"))
        meta   = json.load(open(f"{MODELS_DIR}/metadata.json",    "r"))

        feat_extractor = tf.keras.Model(
            inputs=cnn.input,
            outputs=cnn.layers[-4].output
        )
        return cnn, svm, rf, knn, scaler, feat_extractor, meta, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)


def preprocess_image(img: Image.Image):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img_rgb = img.convert("RGB").resize(IMG_SIZE)
    arr     = np.array(img_rgb, dtype=np.float32)
    arr     = preprocess_input(arr)
    return np.expand_dims(arr, 0)


@st.cache_resource(show_spinner=False)
def load_imagenet_classifier():
    """
    Load MobileNetV2 pre-trained on ImageNet as a zero-cost leaf gate.
    This model is already part of TensorFlow — no extra download needed
    beyond the standard ImageNet weights (~14 MB, cached after first run).
    """
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
    mobilenet = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))
    mobilenet.trainable = False
    return mobilenet


# Full set of ImageNet label keywords that map to plant / leaf / nature content.
# Any top-5 prediction whose label contains one of these words is counted as vegetation.
_PLANT_KEYWORDS = {
    # Explicit leaf / plant labels in ImageNet
    "leaf", "leaves", "plant", "herb", "fern", "frond", "algae",
    "moss", "lichen", "shrub", "bush", "tree", "flower", "blossom",
    "petal", "sepal", "stamen", "pistil", "bud", "shoot", "stem",
    "vine", "creeper", "ivy", "palm", "bamboo", "grass", "reed",
    "corn", "wheat", "rice", "crop", "garden", "greenhouse",
    # Citrus-adjacent
    "lemon", "lime", "orange", "citrus", "grapefruit", "tangerine",
    "mandarin", "pomelo", "kumquat",
    # Other fruits whose leaves look similar and are acceptable
    "mango", "guava", "avocado", "papaya", "banana", "fig",
    "mulberry", "jackfruit", "breadfruit",
    # Common ImageNet plant class names
    "daisy", "sunflower", "rose", "tulip", "artichoke", "broccoli",
    "cauliflower", "cabbage", "lettuce", "spinach", "kale",
    "cucumber", "zucchini", "squash", "pumpkin",
    # Nature / outdoor scenes that often contain heavy vegetation
    "rainforest", "jungle", "forest", "woodland",
}

# ImageNet labels that are definitively NOT leaves — hard-reject if top-1 is here
_HARD_REJECT_KEYWORDS = {
    "person", "man", "woman", "boy", "girl", "child", "face", "head",
    "human", "people", "crowd",
    "dog", "cat", "bird", "fish", "insect", "snake", "lizard", "reptile",
    "animal", "mammal",
    "car", "truck", "bus", "vehicle", "motorcycle", "bicycle", "train",
    "airplane", "ship", "boat",
    "building", "house", "tower", "bridge", "road", "street", "sidewalk",
    "sky", "cloud", "ocean", "river", "lake", "beach", "sand", "rock",
    "mountain", "snow",
    "food", "pizza", "burger", "sandwich", "bread", "cake", "cookie",
    "plate", "bowl", "cup", "bottle", "glass", "utensil",
    "phone", "laptop", "computer", "keyboard", "screen", "television",
    "book", "paper", "document", "sign", "text",
    "furniture", "chair", "table", "sofa", "bed",
    "x-ray", "radiograph", "medical",
}


def is_leaf_image(img: Image.Image) -> tuple[bool, str]:
    """
    Two-stage leaf validator:

    Stage 1 — MobileNetV2 ImageNet classifier (primary, semantic check)
      Runs the image through a pre-trained ImageNet model and inspects the
      top-5 predicted labels. If any are plant/leaf keywords the image passes.
      If the top-1 label is a hard-reject keyword (person, car, food…) it fails
      immediately with a descriptive message.

    Stage 2 — Pixel heuristics (safety net)
      If the classifier cannot confidently classify (all top-5 below 5 %
      confidence), fall back to colour and texture checks so the app still
      works even on unusual images.

    Returns: (is_valid: bool, reason: str)
    """
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input, decode_predictions
    )

    try:
        mobilenet = load_imagenet_classifier()

        # Prepare input — MobileNetV2 expects 224×224 RGB
        img_resized = img.convert("RGB").resize((224, 224))
        arr = np.array(img_resized, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = mobilenet.predict(arr, verbose=0)
        # decode_predictions returns list of (class_id, label, score) tuples
        top5 = decode_predictions(preds, top=5)[0]  # [(id, label, score), ...]

        top1_label = top5[0][1].lower().replace("_", " ")
        top1_score = float(top5[0][2])

        # ── Hard reject on top-1 if it's clearly not a plant ──────
        for reject_kw in _HARD_REJECT_KEYWORDS:
            if reject_kw in top1_label and top1_score > 0.15:
                # Give a user-friendly name for what was detected
                detected = top1_label.replace("_", " ").title()
                return False, (
                    f"This image appears to show '{detected}', not a citrus leaf. "
                    f"Please upload a clear photo of a citrus leaf."
                )

        # ── Check all top-5 labels for plant/leaf keywords ────────
        plant_score = 0.0
        for _, label, score in top5:
            label_clean = label.lower().replace("_", " ")
            for kw in _PLANT_KEYWORDS:
                if kw in label_clean:
                    plant_score += float(score)
                    break  # count each prediction once

        # If cumulative plant probability across top-5 is meaningful → pass
        if plant_score >= 0.08:
            return True, "ok"

        # ── Low-confidence case: classifier unsure ─────────────────
        # Fall through to pixel heuristics below
        if top1_score < 0.10:
            raise ValueError("low_confidence")

        # Classifier is confident but predicted non-plant content
        detected = top1_label.replace("_", " ").title()
        return False, (
            f"No leaf or plant detected. The image appears to show '{detected}'. "
            f"Please upload a citrus leaf photo."
        )

    except Exception as e:
        # ── Stage 2: pixel-level fallback ─────────────────────────
        # Used when: model not loaded, low confidence, or any other error
        img_rgb  = np.array(img.convert("RGB"),  dtype=np.float32) / 255.0
        img_gray = np.array(img.convert("L"),    dtype=np.float32)

        R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

        mean_bright = float(np.mean(img_gray))
        if mean_bright < 12 or mean_bright > 245:
            return False, "Image is too dark or overexposed. Please upload a clear leaf photo."

        std_bright = float(np.std(img_gray))
        if std_bright < 14:
            return False, "Image appears to be a solid colour with no leaf detail."

        # Vegetation colour ratio
        green_mask  = (G > R * 0.78) & (G > B * 0.78) & (G > 0.08)
        yellow_mask = (R > 0.22) & (G > 0.22) & (B < 0.42) & (G >= R * 0.60)
        brown_mask  = (R > 0.22) & (G > 0.14) & (B < 0.26) & (R > G * 0.78) & (G > B * 1.08)
        veg_ratio   = (
            float(green_mask.mean()) +
            float(yellow_mask.mean()) * 0.65 +
            float(brown_mask.mean())  * 0.35
        )
        if veg_ratio < 0.10:
            return False, "No leaf or plant colour detected. Please upload a citrus leaf photo."

        # Saturation check — reject near-greyscale
        pmax = np.max(img_rgb, axis=2)
        pmin = np.min(img_rgb, axis=2)
        sat  = np.where(pmax > 0, (pmax - pmin) / (pmax + 1e-6), 0.0)
        if float(np.mean(sat > 0.15)) < 0.05:
            return False, "Image appears greyscale or monochrome — not a leaf photo."

        # Colour balance — flag red or blue dominance
        if float(np.mean(R)) > float(np.mean(G)) * 1.5 and float(np.mean(R)) > 0.28:
            return False, "Image appears to be a red object or face, not a leaf."
        if float(np.mean(B)) > float(np.mean(G)) * 1.5 and float(np.mean(B)) > 0.26:
            return False, "Image appears to be sky, water, or a blue surface, not a leaf."

        return True, "ok"


def run_prediction(img, cnn, svm, rf, knn, scaler, feat_extractor):
    batch = preprocess_image(img)

    t0 = time.time()
    cnn_proba = cnn.predict(batch, verbose=0)[0]
    cnn_time  = (time.time() - t0) * 1000

    feats    = feat_extractor.predict(batch, verbose=0)
    feats_sc = scaler.transform(feats)

    t0 = time.time()
    svm_proba = svm.predict_proba(feats_sc)[0]
    svm_time  = (time.time() - t0) * 1000

    t0 = time.time()
    rf_proba  = rf.predict_proba(feats_sc)[0]
    rf_time   = (time.time() - t0) * 1000

    t0 = time.time()
    knn_proba = knn.predict_proba(feats_sc)[0]
    knn_time  = (time.time() - t0) * 1000

    models_out = {
        "CNN":           {"proba": cnn_proba, "time_ms": cnn_time},
        "SVM":           {"proba": svm_proba, "time_ms": svm_time},
        "Random Forest": {"proba": rf_proba,  "time_ms": rf_time},
        "KNN":           {"proba": knn_proba, "time_ms": knn_time},
    }
    for name, d in models_out.items():
        idx = int(np.argmax(d["proba"]))
        d["class_idx"]  = idx
        d["class_name"] = CLASS_LABELS[idx]
        d["class_key"]  = CLASSES[idx]
        d["confidence"] = float(d["proba"][idx])
        d["all_proba"]  = {CLASS_LABELS[i]: float(d["proba"][i]) for i in range(4)}

    votes     = [d["class_key"] for d in models_out.values()]
    consensus = max(set(votes), key=votes.count)
    return models_out, consensus


# ── Professional PDF Report ──────────────────────────────────────────
def make_pdf_report(img, models_out, consensus, meta):
    """Generate a clean, professional PDF report."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from io import BytesIO

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=18*mm,
        bottomMargin=18*mm,
    )

    # Color palette
    C_GREEN  = colors.HexColor("#2a8a38")
    C_DARK   = colors.HexColor("#1a2a1a")
    C_GRAY   = colors.HexColor("#4a6a4a")
    C_LGRAY  = colors.HexColor("#8aaa8a")
    C_BG     = colors.HexColor("#f4f7f4")
    C_BORDER = colors.HexColor("#d0ddd0")
    C_GOLD   = colors.HexColor("#b07820")
    C_RED    = colors.HexColor("#c43030")
    C_ORANGE = colors.HexColor("#c05820")
    C_PURPLE = colors.HexColor("#6a3ec0")
    C_TEAL   = colors.HexColor("#1a8878")

    cons_info = DISEASE_INFO.get(consensus, {})
    sev = cons_info.get("severity", "medium")
    sev_color = {"high": C_RED, "medium": C_ORANGE, "low": C_GREEN}[sev]
    dis_color = colors.HexColor(cons_info.get("color", "#2a8a38"))

    styles = getSampleStyleSheet()

    def S(name, **kwargs):
        return ParagraphStyle(name, **kwargs)

    style_h1 = S("H1", fontSize=22, fontName="Helvetica-Bold", textColor=C_DARK,
                  leading=26, spaceAfter=2)
    style_h2 = S("H2", fontSize=13, fontName="Helvetica-Bold", textColor=C_DARK,
                  leading=16, spaceBefore=10, spaceAfter=4)
    style_h3 = S("H3", fontSize=10, fontName="Helvetica-Bold", textColor=C_GRAY,
                  leading=13, spaceBefore=6, spaceAfter=2)
    style_body = S("Body", fontSize=9.5, fontName="Helvetica", textColor=C_GRAY,
                   leading=14, spaceAfter=2)
    style_small = S("Small", fontSize=8, fontName="Helvetica", textColor=C_LGRAY,
                    leading=11)
    style_label = S("Label", fontSize=7.5, fontName="Helvetica-Bold", textColor=C_LGRAY,
                    leading=10, spaceAfter=1)
    style_accent = S("Accent", fontSize=9, fontName="Helvetica-Bold", textColor=C_GREEN,
                     leading=12)
    style_center = S("Center", fontSize=9.5, fontName="Helvetica", textColor=C_GRAY,
                     leading=14, alignment=TA_CENTER)

    story = []

    # ── Header ──
    now_str = datetime.now().strftime("%d %B %Y at %H:%M")
    header_data = [[
        Paragraph("<b>CitrusMD</b>", S("BoldGreen", fontSize=20, fontName="Helvetica-Bold",
                                       textColor=C_GREEN, leading=24)),
        Paragraph(f"Disease Detection Report<br/><font size='8' color='#8aaa8a'>Generated {now_str}</font>",
                  S("R", fontSize=10, fontName="Helvetica", textColor=C_GRAY,
                    leading=13, alignment=TA_RIGHT))
    ]]
    header_tbl = Table(header_data, colWidths=["50%", "50%"])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(header_tbl)
    story.append(HRFlowable(width="100%", thickness=1.2, color=C_GREEN, spaceAfter=14))

    # ── Consensus banner ──
    story.append(Paragraph("DIAGNOSIS RESULT", style_label))
    story.append(Spacer(1, 3))

    votes = [d["class_key"] for d in models_out.values()]
    agree_count = votes.count(consensus)

    cons_table_data = [[
        Paragraph(
            f"<b>{cons_info.get('display', consensus)}</b>",
            S("Dis", fontSize=18, fontName="Helvetica-Bold", textColor=dis_color, leading=22)
        ),
        Paragraph(
            f"<b>{agree_count}/4</b><br/><font size='8'>models agree</font>",
            S("Agree", fontSize=13, fontName="Helvetica-Bold", textColor=C_GREEN,
              leading=16, alignment=TA_CENTER)
        ),
    ]]
    cons_tbl = Table(cons_table_data, colWidths=["75%", "25%"])
    cons_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG),
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("LINEABOVE", (0, 0), (-1, 0), 3, dis_color),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 14),
        ("RIGHTPADDING", (-1, 0), (-1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(cons_tbl)
    story.append(Spacer(1, 4))

    # Cause + severity row
    sev_label = {"high": "HIGH SEVERITY", "medium": "MEDIUM SEVERITY", "low": "LOW SEVERITY"}[sev]
    meta2_data = [[
        Paragraph(f"<b>Cause:</b> {cons_info.get('cause', 'N/A')}", style_body),
        Paragraph(f"<b>{sev_label}</b>",
                  S("SevL", fontSize=8.5, fontName="Helvetica-Bold",
                    textColor=sev_color, leading=11, alignment=TA_RIGHT)),
    ]]
    meta2_tbl = Table(meta2_data, colWidths=["68%", "32%"])
    meta2_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta2_tbl)
    story.append(Paragraph(f"<b>Symptoms:</b> {cons_info.get('symptoms', '')}", style_body))
    story.append(Spacer(1, 12))

    # ── Input image ──
    story.append(Paragraph("INPUT IMAGE", style_label))
    story.append(Spacer(1, 3))
    img_buf = BytesIO()
    img.convert("RGB").resize((260, 260)).save(img_buf, format="PNG")
    img_buf.seek(0)
    rl_img = RLImage(img_buf, width=60*mm, height=60*mm)
    img_tbl = Table([[rl_img]], colWidths=[65*mm])
    img_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("BACKGROUND", (0, 0), (-1, -1), C_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 14))

    # ── Per-model predictions ──
    story.append(Paragraph("MODEL PREDICTIONS", style_label))
    story.append(Spacer(1, 4))

    MODEL_COLORS_MAP = {
        "CNN": colors.HexColor("#2a8a38"),
        "SVM": colors.HexColor("#1a8878"),
        "Random Forest": colors.HexColor("#c05820"),
        "KNN": colors.HexColor("#6a3ec0"),
    }

    pred_header = [
        Paragraph("<b>Model</b>", S("TH", fontSize=8.5, fontName="Helvetica-Bold",
                                    textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Prediction</b>", S("TH", fontSize=8.5, fontName="Helvetica-Bold",
                                         textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Confidence</b>", S("TH", fontSize=8.5, fontName="Helvetica-Bold",
                                          textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Canker</b>", S("TH", fontSize=8, fontName="Helvetica-Bold",
                                      textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Greening</b>", S("TH", fontSize=8, fontName="Helvetica-Bold",
                                        textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Black Spot</b>", S("TH", fontSize=8, fontName="Helvetica-Bold",
                                          textColor=C_DARK, leading=11, alignment=TA_CENTER)),
        Paragraph("<b>Nutrient Def.</b>", S("TH", fontSize=8, fontName="Helvetica-Bold",
                                             textColor=C_DARK, leading=11, alignment=TA_CENTER)),
    ]
    pred_rows = [pred_header]
    for mname, d in models_out.items():
        conf_pct = f"{d['confidence']*100:.1f}%"
        row = [
            Paragraph(f"<b>{mname}</b>",
                      S("MN", fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=MODEL_COLORS_MAP.get(mname, C_GREEN), leading=11)),
            Paragraph(d["class_name"],
                      S("DN", fontSize=8.5, fontName="Helvetica", textColor=C_DARK, leading=11)),
            Paragraph(f"<b>{conf_pct}</b>",
                      S("CF", fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=C_GREEN if d["confidence"] > 0.7 else C_ORANGE, leading=11,
                        alignment=TA_CENTER)),
        ]
        for cl in CLASS_LABELS:
            p = d["all_proba"][cl]
            row.append(Paragraph(f"{p*100:.1f}%",
                                  S("P", fontSize=8, fontName="Helvetica",
                                    textColor=C_DARK if p > 0.3 else C_LGRAY,
                                    leading=11, alignment=TA_CENTER)))
        pred_rows.append(row)

    pred_tbl = Table(pred_rows, colWidths=[28*mm, 32*mm, 24*mm, 20*mm, 20*mm, 22*mm, 25*mm])
    pred_ts = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4ece4")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    pred_tbl.setStyle(TableStyle(pred_ts))
    story.append(pred_tbl)
    story.append(Spacer(1, 14))

    # ── Treatment recommendations ──
    story.append(Paragraph("TREATMENT RECOMMENDATIONS", style_label))
    story.append(Spacer(1, 4))

    recs = cons_info.get("recommendations", [])
    rec_rows = []
    for i, rec in enumerate(recs):
        rec_rows.append([
            Paragraph(f"{i+1}.", S("Num", fontSize=9, fontName="Helvetica-Bold",
                                    textColor=C_GREEN, leading=13, alignment=TA_CENTER)),
            Paragraph(rec, S("RecTxt", fontSize=9, fontName="Helvetica",
                              textColor=C_DARK, leading=14)),
        ])

    rec_tbl = Table(rec_rows, colWidths=[10*mm, None])
    rec_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_BG),
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("LINEABOVE", (0, 0), (-1, 0), 2.5, C_GOLD),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, C_BG]),
    ]))
    story.append(rec_tbl)
    story.append(Spacer(1, 14))

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=0.8, color=C_BORDER, spaceBefore=4, spaceAfter=6))
    story.append(Paragraph(
        "This report is generated by CitrusMD for advisory purposes only. "
        "Always consult a qualified agronomist or plant pathologist before applying treatments. "
        "Bells University of Technology — B.Tech Information Technology — Citrus Disease Classification Project.",
        style_small
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ──────────────────────────────────────────────────────────
def render_sidebar(meta):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sb-logo">
            <div class="sb-logo-title">Citrus<span>MD</span></div>
            <div class="sb-logo-sub">Citrus Disease Detection</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
        page = st.radio(
            "nav",
            ["Diagnose", "Performance", "Models", "Disease Guide"],
            label_visibility="collapsed",
        )

        # Theme toggle
        st.markdown('<div class="sb-section" style="margin-top:0.8rem;">Theme</div>', unsafe_allow_html=True)
        st.markdown('<div class="theme-btn-wrap">', unsafe_allow_html=True)
        label = "Switch to Light Mode" if dark else "Switch to Dark Mode"
        if st.button(label, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        if meta:
            st.markdown('<div class="sb-section" style="margin-top:1rem;">Model Accuracy</div>',
                        unsafe_allow_html=True)

            model_colors = {
                "CNN": "#4ecb5e",
                "SVM": "#3dc0a8",
                "Random_Forest": "#f08040",
                "KNN": "#9b6aef",
            }
            display_names = {
                "CNN": "CNN",
                "SVM": "SVM",
                "Random_Forest": "Random Forest",
                "KNN": "KNN",
            }
            for mk, col in model_colors.items():
                if mk in meta:
                    acc = meta[mk].get("accuracy", 0)
                    pct = int(acc * 100)
                    dn  = display_names[mk]
                    st.markdown(f"""
                    <div class="sb-acc-row">
                        <div class="sb-acc-top">
                            <span class="sb-acc-name">{dn}</span>
                            <span class="sb-acc-val" style="color:{col};">{pct}%</span>
                        </div>
                        <div class="sb-acc-bar">
                            <div class="sb-acc-fill" style="width:{pct}%; background:{col};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sb-footer">
            <div class="sb-footer-text">
                Bells University of Technology<br>
                B.Tech Information Technology<br>
                Citrus Disease Classification
            </div>
        </div>
        """, unsafe_allow_html=True)

    return page


# ── Pages ────────────────────────────────────────────────────────────
def page_detection(cnn, svm, rf, knn, scaler, feat_extractor, meta):
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Leaf Analysis</div>
        <div class="page-header-title">Citrus <span>Disease Detection</span></div>
        <div class="page-header-sub">
            Upload a citrus leaf photo. See classification across CNN, SVM, Random Forest and KNN models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload citrus leaf",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption="")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-val">{img.size[0]}x{img.size[1]}</div>
                    <div class="stat-lbl">Resolution</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-val">{uploaded.size/1024:.0f} KB</div>
                    <div class="stat-lbl">File Size</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            go = st.button("Run Diagnosis", use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-placeholder">
                <div class="upload-placeholder-icon">🍃</div>
                <div class="upload-placeholder-text">No image uploaded yet</div>
            </div>
            """, unsafe_allow_html=True)
            go = False

    with col_right:
        if uploaded and go:
            img = Image.open(uploaded)
            with st.spinner("Validating image — checking for citrus leaf content..."):
                valid, reason = is_leaf_image(img)

            if not valid:
                st.markdown(f"""
                <div class="warn-card">
                    <div class="warn-title">Not a Citrus Leaf</div>
                    <div class="warn-sub">{reason}<br><br>
                    Please upload a clear photo of a citrus leaf. The leaf should be visible,
                    well-lit, and fill most of the frame.</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-label">Tips for a Good Photo</div>', unsafe_allow_html=True)
                tips = [
                    "Photograph a single citrus leaf against a plain or natural background",
                    "Ensure the leaf fills most of the frame",
                    "Use good, even lighting — avoid heavy shadows or glare",
                    "Both the top surface and underside of the leaf are accepted",
                    "JPG, PNG, BMP and WEBP formats are supported",
                ]
                for t in tips:
                    st.markdown(f"""<div class="rec-item">
                        <span class="rec-bullet">—</span>
                        <span>{t}</span>
                    </div>""", unsafe_allow_html=True)

            else:
                with st.spinner("Running diagnosis across all 4 models..."):
                    models_out, consensus = run_prediction(img, cnn, svm, rf, knn, scaler, feat_extractor)

                cons_info  = DISEASE_INFO.get(consensus, {})
                votes      = [d["class_key"] for d in models_out.values()]
                agree      = votes.count(consensus)

                # Consensus card
                st.markdown(f"""
                <div class="consensus-card" style="border-top-color:{cons_info.get('color','#4ecb5e')};">
                    <div class="cons-label">Consensus Diagnosis</div>
                    <div class="cons-name">{cons_info.get('display', consensus)}</div>
                    <div class="cons-cause">{cons_info.get('cause', '')}</div>
                    <div class="cons-agree">&#10003; {agree} of 4 models agree</div>
                </div>
                """, unsafe_allow_html=True)

                # Model predictions
                st.markdown('<div class="section-label">Individual Model Predictions</div>',
                            unsafe_allow_html=True)

                MODEL_COLORS = {
                    "CNN": "#4ecb5e",
                    "SVM": "#3dc0a8",
                    "Random Forest": "#f08040",
                    "KNN": "#9b6aef",
                }
                for mname, d in models_out.items():
                    conf_pct = int(d["confidence"] * 100)
                    dis_color = DISEASE_INFO.get(d["class_key"], {}).get("color", "#4ecb5e")
                    m_col = MODEL_COLORS.get(mname, "#4ecb5e")
                    st.markdown(f"""
                    <div class="pred-card">
                        <div class="pred-accent-bar" style="background:{m_col};"></div>
                        <div style="flex:1; min-width:0;">
                            <div class="pred-model-name">{mname}</div>
                            <div class="pred-disease-name">{d['class_name']}</div>
                            <div class="pred-conf-row">
                                <span>{conf_pct}% confidence</span>
                                <div class="conf-track">
                                    <div class="conf-fill" style="width:{conf_pct}%; background:{dis_color};"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Severity + recommendations
                sev       = cons_info.get("severity", "medium")
                sev_class = {"high": "chip-high", "medium": "chip-medium", "low": "chip-low"}[sev]
                sev_label = {"high": "High Severity", "medium": "Medium Severity", "low": "Low Severity"}[sev]

                st.markdown('<div class="section-label">Treatment Recommendations</div>',
                            unsafe_allow_html=True)
                st.markdown(f"<span class='chip {sev_class}'>{sev_label}</span>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-header">{cons_info.get('display', '')} — Recommended Actions</div>
                """, unsafe_allow_html=True)
                for r in cons_info.get("recommendations", []):
                    st.markdown(f"""
                    <div class="rec-item">
                        <span class="rec-bullet">•</span>
                        <span>{r}</span>
                    </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Export
                st.markdown('<div class="section-label">Export Report</div>', unsafe_allow_html=True)
                try:
                    pdf = make_pdf_report(img, models_out, consensus, meta)
                    st.download_button(
                        "Download PDF Report",
                        data=pdf,
                        file_name=f"CitrusMD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except ImportError:
                    # Fallback if reportlab not installed
                    try:
                        pdf = make_pdf_report_mpl(img, models_out, consensus, meta)
                        st.download_button(
                            "Download PDF Report",
                            data=pdf,
                            file_name=f"CitrusMD_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"PDF generation error: {e}. Install reportlab: pip install reportlab")
                except Exception as e:
                    st.warning(f"PDF error: {e}")

        elif not uploaded:
            st.markdown("""
            <div class="upload-placeholder" style="height:420px;">
                <div class="upload-placeholder-icon" style="font-size:3rem;">🍋</div>
                <div class="upload-placeholder-text">Upload an image to begin</div>
                <div style="font-size:0.78rem; color:var(--text4);">CNN · SVM · Random Forest · KNN</div>
            </div>
            """, unsafe_allow_html=True)


def make_pdf_report_mpl(img, models_out, consensus, meta):
    """Fallback matplotlib PDF if reportlab unavailable."""
    cons_info = DISEASE_INFO.get(consensus, {})
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#ffffff")

    ax = fig.add_axes([0.08, 0.94, 0.84, 0.04])
    ax.axis("off")
    ax.text(0, 0.5, "CitrusMD — Disease Detection Report",
            fontsize=16, fontweight="bold", color="#2a8a38", va="center")
    ax.text(1, 0.5, datetime.now().strftime("%d %B %Y"),
            fontsize=9, color="#8aaa8a", va="center", ha="right")

    ax2 = fig.add_axes([0.08, 0.88, 0.84, 0.005])
    ax2.axis("off")
    ax2.axhline(0.5, color="#2a8a38", linewidth=2)

    ax3 = fig.add_axes([0.08, 0.76, 0.22, 0.11])
    ax3.imshow(np.array(img.convert("RGB").resize((180, 180))))
    ax3.axis("off")
    ax3.set_title("Input", fontsize=8, color="#4a6a4a", pad=3)

    ax4 = fig.add_axes([0.33, 0.76, 0.58, 0.11], facecolor="#f4f7f4")
    ax4.axis("off")
    ax4.text(0.02, 0.85, "DIAGNOSIS", fontsize=7, fontweight="bold", color="#8aaa8a",
             transform=ax4.transAxes)
    ax4.text(0.02, 0.55, cons_info.get("display", consensus),
             fontsize=16, fontweight="bold", color=cons_info.get("color", "#2a8a38"),
             transform=ax4.transAxes)
    ax4.text(0.02, 0.25, cons_info.get("cause", ""),
             fontsize=8, color="#4a6a4a", transform=ax4.transAxes)
    votes = [d["class_key"] for d in models_out.values()]
    ax4.text(0.02, 0.05, f"Model agreement: {votes.count(consensus)}/4",
             fontsize=8, color="#2a8a38", transform=ax4.transAxes)

    ax5 = fig.add_axes([0.08, 0.52, 0.84, 0.21], facecolor="#ffffff")
    ax5.axis("off")
    ax5.set_title("Model Predictions", fontsize=10, fontweight="bold", color="#1a2a1a",
                  loc="left", pad=6)
    headers = ["Model", "Prediction", "Confidence", "Canker", "Greening", "Black Spot", "Nutrient Def."]
    rows = []
    for mn, d in models_out.items():
        rows.append([mn, d["class_name"], f"{d['confidence']*100:.1f}%"] +
                    [f"{d['all_proba'][cl]*100:.1f}%" for cl in CLASS_LABELS])
    tbl = ax5.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#f4f7f4" if row == 0 else ("#ffffff" if row % 2 else "#f8fbf8"))
        cell.set_text_props(color="#1a2a1a" if row > 0 else "#2a8a38",
                            fontweight="bold" if row == 0 else "normal")
        cell.set_edgecolor("#d0ddd0")

    ax6 = fig.add_axes([0.08, 0.08, 0.84, 0.41], facecolor="#f4f7f4")
    ax6.axis("off")
    ax6.set_title("Treatment Recommendations", fontsize=10, fontweight="bold", color="#b07820",
                  loc="left", pad=6)
    for i, rec in enumerate(cons_info.get("recommendations", [])[:6]):
        ax6.text(0.02, 0.88 - i*0.15, f"{i+1}. {rec}",
                 fontsize=8, color="#1a2a1a", va="top", wrap=True, transform=ax6.transAxes)

    buf = io.BytesIO()
    plt.savefig(buf, format="pdf", facecolor="#ffffff", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def page_performance(meta):
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Training Results</div>
        <div class="page-header-title">Model <span>Performance</span></div>
        <div class="page-header-sub">Classification accuracy, precision, recall and F1-score across all models.</div>
    </div>
    """, unsafe_allow_html=True)

    if not meta:
        st.warning("metadata.json not found in models/ folder.")
        return

    model_keys    = ["CNN", "SVM", "Random_Forest", "KNN"]
    display_names = ["CNN", "SVM", "Random Forest", "KNN"]
    colors_list   = ["#4ecb5e", "#3dc0a8", "#f08040", "#9b6aef"]

    # Accuracy tiles
    st.markdown('<div class="section-label">Accuracy Overview</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, mk, dn, color in zip(cols, model_keys, display_names, colors_list):
        if mk in meta:
            acc = meta[mk].get("accuracy", 0)
            pct = acc * 100
            with col:
                st.markdown(f"""
                <div class="perf-tile" style="border-top:3px solid {color};">
                    <div class="perf-model">{dn}</div>
                    <div class="perf-pct" style="color:{color};">{pct:.1f}%</div>
                    <div class="perf-dec">{acc:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
    metrics       = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_bg    = "#0a0f0a" if dark else "#ffffff"
    plot_text  = "#edf2ed" if dark else "#1a2a1a"
    plot_grid  = "#223022" if dark else "#d0ddd0"
    plot_sub   = "#8aaa8a" if dark else "#4a6a4a"

    values = np.array([[meta[mk].get(m, 0) for m in metrics] for mk in model_keys if mk in meta])

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=plot_bg)
    ax.set_facecolor(plot_bg)
    x = np.arange(4)
    w = 0.18
    for i, (dn, color, vals) in enumerate(zip(display_names, colors_list, values)):
        bars = ax.bar(x + i * w, vals, w, label=dn, color=color, alpha=0.9)
        for bar in bars:
            ax.text(bar.get_x() + w / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7, color=plot_text)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(metric_labels, color=plot_text, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color=plot_sub, fontsize=9)
    ax.set_title("Classification Metrics — All 4 Models", color=plot_text, fontsize=12,
                 fontweight="bold", pad=10)
    ax.legend(facecolor=plot_bg, labelcolor=plot_text, framealpha=0.9,
              edgecolor=plot_grid, fontsize=9)
    ax.tick_params(colors=plot_sub)
    ax.grid(axis="y", alpha=0.15, color=plot_grid)
    for sp in ax.spines.values():
        sp.set_edgecolor(plot_grid)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Metrics table
    st.markdown('<div class="section-label">Full Metrics Table</div>', unsafe_allow_html=True)
    import pandas as pd
    rows = []
    for mk, dn in zip(model_keys, display_names):
        if mk in meta:
            m = meta[mk]
            rows.append({
                "Model":     dn,
                "Accuracy":  f"{m.get('accuracy', 0)*100:.2f}%",
                "Precision": f"{m.get('precision', 0)*100:.2f}%",
                "Recall":    f"{m.get('recall', 0)*100:.2f}%",
                "F1-Score":  f"{m.get('f1_score', 0)*100:.2f}%",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_models():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Architecture</div>
        <div class="page-header-title">About the <span>Models</span></div>
        <div class="page-header-sub">How each of the 4 classifiers works and where it excels.</div>
    </div>
    """, unsafe_allow_html=True)

    models_info = [
        {
            "name": "Convolutional Neural Network (CNN)",
            "sub": "MobileNetV2 — Transfer Learning",
            "color": "#4ecb5e",
            "desc": "The CNN learns hierarchical spatial features from raw pixels — edges in early layers and disease patterns in deeper layers. MobileNetV2 pre-trained on ImageNet is fine-tuned on citrus disease images in two phases.",
            "strengths": [
                "Automatic feature extraction from raw pixels",
                "Transfer learning reduces training data requirements",
                "Captures complex disease lesion patterns",
                "Highest accuracy across all classes",
            ],
            "limits": [
                "Requires more compute for inference",
                "Lower interpretability than classical models",
                "Needs more labelled data than classical methods",
            ],
        },
        {
            "name": "Support Vector Machine (SVM)",
            "sub": "RBF Kernel — Grid-Search Tuned",
            "color": "#3dc0a8",
            "desc": "SVM finds an optimal hyperplane maximising the margin between classes in the 1280-dimensional MobileNetV2 feature space. The RBF kernel enables non-linear decision boundaries.",
            "strengths": [
                "Strong generalisation via margin maximisation",
                "Effective in high-dimensional feature spaces",
                "Fast inference once trained",
            ],
            "limits": [
                "Slow training with large datasets and Grid Search",
                "Sensitive to feature scaling",
                "Less accurate than CNN on complex visual patterns",
            ],
        },
        {
            "name": "Random Forest",
            "sub": "Ensemble of 100–300 Decision Trees",
            "color": "#f08040",
            "desc": "Builds many independent decision trees on bootstrap samples, each using a random feature subset per split. Final prediction is a majority vote — reducing variance compared to a single tree.",
            "strengths": [
                "Robust to overfitting due to ensemble averaging",
                "Provides feature importance scores",
                "Fast at inference time",
            ],
            "limits": [
                "Lower accuracy than CNN on image-based features",
                "Memory-intensive for large forests",
                "Many trees needed for stability",
            ],
        },
        {
            "name": "K-Nearest Neighbours (KNN)",
            "sub": "Lazy Learner — Euclidean Distance",
            "color": "#9b6aef",
            "desc": "KNN stores all training embeddings and classifies new inputs by finding the K closest training points in feature space. K is chosen via 5-fold cross-validation. No explicit model training is required.",
            "strengths": [
                "No training phase required",
                "Non-parametric — no distributional assumptions",
                "Decisions are highly interpretable",
            ],
            "limits": [
                "Slow inference on large training sets",
                "Sensitive to choice of K and feature scaling",
                "Struggles with noisy or high-dimensional spaces",
            ],
        },
    ]

    for m in models_info:
        st.markdown(f"""
        <div class="about-card" style="border-left: 3px solid {m['color']};">
            <div style="margin-bottom:0.3rem;">
                <div class="about-title">{m['name']}</div>
                <div class="about-sub" style="color:{m['color']};">{m['sub']}</div>
            </div>
            <div class="about-desc">{m['desc']}</div>
            <div class="about-grid">
                <div>
                    <div class="about-col-title" style="color:{m['color']};">Strengths</div>
                    {''.join(f'<div class="about-item">+ {s}</div>' for s in m['strengths'])}
                </div>
                <div>
                    <div class="about-col-title" style="color:var(--orange);">Limitations</div>
                    {''.join(f'<div class="about-item">- {l}</div>' for l in m['limits'])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def page_diseases():
    st.markdown("""
    <div class="page-header">
        <div class="page-header-eyebrow">Reference</div>
        <div class="page-header-title">Citrus <span>Disease Guide</span></div>
        <div class="page-header-sub">Detailed information on each of the 4 detected conditions.</div>
    </div>
    """, unsafe_allow_html=True)

    for key, info in DISEASE_INFO.items():
        sev   = info["severity"]
        sev_c = {"high": "chip-high", "medium": "chip-medium", "low": "chip-low"}[sev]
        sev_l = {"high": "High Severity", "medium": "Medium Severity", "low": "Low Severity"}[sev]

        st.markdown(f"""
        <div class="disease-card" style="border-left: 3px solid {info['color']};">
            <div style="display:flex; align-items:flex-start; justify-content:space-between;
                        flex-wrap:wrap; gap:0.5rem; margin-bottom:0.6rem;">
                <div>
                    <div class="disease-name" style="color:{info['color']};">{info['display']}</div>
                    <div class="disease-cause">{info['cause']}</div>
                </div>
                <span class="chip {sev_c}">{sev_l}</span>
            </div>
            <div class="symptom-box">
                <div class="symp-label">Symptoms</div>
                {info['symptoms']}
            </div>
            <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
                        letter-spacing:0.15em; color:var(--gold); margin-bottom:0.5rem;">
                Recommendations
            </div>
            {''.join(f'<div class="rec-item"><span class="rec-bullet">•</span><span>{r}</span></div>'
                     for r in info['recommendations'])}
        </div>
        """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    cnn, svm, rf, knn, scaler, feat_extractor, meta, err = load_all_models()
    models_loaded = cnn is not None

    if not models_loaded:
        st.markdown("""
        <div class="page-header">
            <div class="page-header-eyebrow">Setup Required</div>
            <div class="page-header-title">CitrusMD <span>Setup Guide</span></div>
            <div class="page-header-sub">Train the models first, then place them in the <code>models/</code> folder.</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card" style="border-left:3px solid var(--red);">
            <div class="about-title" style="color:var(--red);">Model Files Not Found</div>
            <div class="about-desc">The app requires trained model files. Follow the steps below to set up.</div>
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "Run the Google Colab notebook",
             "Complete all training cells. The final cell downloads a ZIP file."),
            ("2", "Extract the ZIP file",
             "You will get: cnn_model.h5, svm_model.pkl, rf_model.pkl, knn_model.pkl, scaler.pkl, metadata.json"),
            ("3", "Create the models/ folder",
             "Create a folder named models in the same directory as app.py"),
            ("4", "Place files inside models/",
             "Copy all 6 files into the models/ folder"),
            ("5", "Restart the app",
             "Run: streamlit run app.py"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="setup-step">
                <div class="setup-num">{num}</div>
                <div>
                    <div class="setup-title">{title}</div>
                    <div class="setup-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if err:
            st.markdown(f"""
            <div class="about-card" style="border-left:3px solid var(--red); margin-top:1rem;">
                <div style="font-size:0.68rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.1em; color:var(--red); margin-bottom:0.4rem;">Error Detail</div>
                <code style="font-size:0.82rem; color:var(--text2);">{err}</code>
            </div>
            """, unsafe_allow_html=True)
        return

    page = render_sidebar(meta)

    if page == "Diagnose":
        page_detection(cnn, svm, rf, knn, scaler, feat_extractor, meta)
    elif page == "Performance":
        page_performance(meta)
    elif page == "Models":
        page_models()
    elif page == "Disease Guide":
        page_diseases()


if __name__ == "__main__":
    main()