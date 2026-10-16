from __future__ import annotations

import html
import logging
import math
import re
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.gate.agreement import answers_are_equivalent, extract_answer_key
from src.gate.response_quality import assess_response_quality
from src.gate.config import EXPECTED_FEATURES, GATE_FILE_V5, MODEL_NAME, N_SAMPLES

# torch and ReliabilityGate are imported lazily (first "Check Reliability" click)
# so the first page render never waits on torch/transformers.
torch = None


logger = logging.getLogger(__name__)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Local LLM Reliability Gate",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONSTANTS
# ============================================================

# Display-only fallback; the live threshold always comes from the gate file.
DEFAULT_THRESHOLD = 0.82

EXAMPLE_QUESTIONS = [
    "What is the capital of Australia?",
    "Who discovered gravity?",
    "What is 15% of 240?",
    "Who wrote Hamlet?",
]

HISTORY_LIMIT = 5

METRIC_NOTES = {
    "agreement": (
        "How often the sampled answers converged on the same "
        "normalized answer."
    ),
    "margin": (
        "Difference between the winning and second-most common answer."
    ),
    "entropy": (
        "Higher values indicate more disagreement among answer candidates."
    ),
    "mean_lp": (
        "Average model log-probability across generated tokens "
        "(closer to 0 means more confident)."
    ),
    "min_lp": (
        "Lowest-confidence token in each answer, averaged across samples."
    ),
}


# ============================================================
# STYLES
# ============================================================

THEMES = {
    "daylight": {
        "label": "Daylight", "--bg": "#f4f7fd", "--surface": "#ffffff",
        "--surface-2": "#eaf0fb", "--border": "#d6e0f2", "--text": "#0b1a33",
        "--muted": "#4a5b78", "--accent": "#2563eb", "--accent-2": "#0369a1",
        "--on-accent": "#ffffff", "--trust": "#047857", "--verify": "#b45309",
        "--error": "#be123c", "--info": "#0369a1",
        "--trust-bg": "rgba(4,120,87,0.08)", "--trust-bd": "rgba(4,120,87,0.28)",
        "--verify-bg": "rgba(180,83,9,0.08)", "--verify-bd": "rgba(180,83,9,0.28)",
        "--error-bg": "rgba(190,18,60,0.08)", "--error-bd": "rgba(190,18,60,0.28)",
        "--info-bg": "rgba(3,105,161,0.08)", "--info-bd": "rgba(3,105,161,0.28)",
        "--track": "#dbe4f5", "--shadow": "0 1px 2px rgba(16,24,40,.06), 0 12px 32px rgba(16,24,40,.08)",
        "--hairline": "rgba(255,255,255,.7)", "--glow-1": "rgba(37,99,235,.10)",
        "--glow-2": "rgba(3,105,161,.08)", "--code-bg": "#eef3fc", "color-scheme": "light",
    },
    "ivory": {
        "label": "Ivory", "--bg": "#faf6ef", "--surface": "#fffdf9",
        "--surface-2": "#f3ece0", "--border": "#e5dccd", "--text": "#241a14",
        "--muted": "#6a5a4d", "--accent": "#86198f", "--accent-2": "#be185d",
        "--on-accent": "#ffffff", "--trust": "#047857", "--verify": "#a15c07",
        "--error": "#b91c1c", "--info": "#0e7490",
        "--trust-bg": "rgba(4,120,87,0.08)", "--trust-bd": "rgba(4,120,87,0.28)",
        "--verify-bg": "rgba(161,92,7,0.08)", "--verify-bd": "rgba(161,92,7,0.28)",
        "--error-bg": "rgba(185,28,28,0.08)", "--error-bd": "rgba(185,28,28,0.28)",
        "--info-bg": "rgba(14,116,144,0.08)", "--info-bd": "rgba(14,116,144,0.28)",
        "--track": "#eadfce", "--shadow": "0 1px 2px rgba(60,40,20,.06), 0 12px 32px rgba(60,40,20,.09)",
        "--hairline": "rgba(255,255,255,.8)", "--glow-1": "rgba(190,24,93,.08)",
        "--glow-2": "rgba(134,25,143,.06)", "--code-bg": "#f6efe3", "color-scheme": "light",
    },
    "midnight": {
        "label": "Midnight", "--bg": "#080c18", "--surface": "#111729",
        "--surface-2": "#182038", "--border": "#26304d", "--text": "#f3efe4",
        "--muted": "#a4abc0", "--accent": "#f2c14e", "--accent-2": "#ffd98a",
        "--on-accent": "#1b1405", "--trust": "#4ade80", "--verify": "#fb923c",
        "--error": "#fb7185", "--info": "#38bdf8",
        "--trust-bg": "rgba(74,222,128,0.12)", "--trust-bd": "rgba(74,222,128,0.35)",
        "--verify-bg": "rgba(251,146,60,0.12)", "--verify-bd": "rgba(251,146,60,0.35)",
        "--error-bg": "rgba(251,113,133,0.12)", "--error-bd": "rgba(251,113,133,0.35)",
        "--info-bg": "rgba(56,189,248,0.12)", "--info-bd": "rgba(56,189,248,0.35)",
        "--track": "#232c47", "--shadow": "0 14px 44px rgba(0,0,0,.5)",
        "--hairline": "rgba(255,255,255,.06)", "--glow-1": "rgba(242,193,78,.10)",
        "--glow-2": "rgba(56,189,248,.06)", "--code-bg": "#0d1324", "color-scheme": "dark",
    },
}


def theme_css(name: str) -> str:
    theme = THEMES[str(name).strip().lower()]
    declarations = "".join(
        f"{token}:{value};" for token, value in theme.items()
        if token.startswith("--")
    )
    return f"<style>:root{{{declarations}color-scheme:{theme['color-scheme']};}}</style>"


CSS = """
<style>
:root {
    --bg: #f4f7fd;
    --surface: #ffffff;
    --surface-2: #eaf0fb;
    --border: #d6e0f2;
    --text: #0b1a33;
    --muted: #4a5b78;
    --accent: #2563eb;
    --accent-2: #0369a1;
    --on-accent: #ffffff;
    --trust: #047857;
    --verify: #b45309;
    --error: #be123c;
    --info: #0369a1;
    --trust-bg: rgba(4,120,87,0.08);
    --trust-bd: rgba(4,120,87,0.28);
    --verify-bg: rgba(180,83,9,0.08);
    --verify-bd: rgba(180,83,9,0.28);
    --error-bg: rgba(190,18,60,0.08);
    --error-bd: rgba(190,18,60,0.28);
    --info-bg: rgba(3,105,161,0.08);
    --info-bd: rgba(3,105,161,0.28);
    --track: #dbe4f5;
    --shadow: 0 1px 2px rgba(16,24,40,.06), 0 12px 32px rgba(16,24,40,.08);
    --hairline: rgba(255,255,255,.7);
    --glow-1: rgba(37,99,235,.10);
    --glow-2: rgba(3,105,161,.08);
    --code-bg: #eef3fc;
    --grad: linear-gradient(135deg, var(--accent), var(--accent-2));
    color-scheme: light;
}

html, body, .stApp {
    color: var(--text);
    color-scheme: inherit;
    font-family: Inter, "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
}
.stApp {
    background:
        radial-gradient(ellipse at 92% 2%, var(--glow-1), transparent 32rem),
        radial-gradient(ellipse at 4% 100%, var(--glow-2), transparent 34rem),
        var(--bg);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="InputInstructions"] { display: none; }
[data-testid="stAppDeployButton"], .stAppDeployButton, .stDeployButton, [data-testid="stDeployButton"] { display: none !important; }
.block-container { position: relative; z-index: 1; max-width: 1180px; padding: .9rem 1.4rem 2rem; }
* { scrollbar-color: var(--border) var(--bg); }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: var(--track); }
*::-webkit-scrollbar-thumb { background: var(--border); border: 2px solid var(--bg); border-radius: 99px; }
::selection { color: var(--text); background: var(--glow-1); }

@keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes dot-pulse { 0%, 100% { box-shadow: 0 0 0 0 var(--verify-bg); } 50% { box-shadow: 0 0 0 4px transparent; } }
@keyframes shimmer { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }
@keyframes think { 0%, 60%, 100% { opacity: .35; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-3px); } }
@keyframes rotate { to { transform: rotate(360deg); } }
@keyframes wordin { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes badgein { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
@keyframes grow { from { width: 0; } to { width: var(--w); } }
@keyframes ringfill { from { stroke-dashoffset: var(--circ); } to { stroke-dashoffset: var(--off); } }
@keyframes pop { from { opacity: 0; transform: scale(.94) translateY(4px); } to { opacity: 1; transform: scale(1) translateY(0); } }

.hero { display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; padding: 0 2px 12px; margin-bottom: 10px; border-bottom: 1px solid var(--border); }
.brand { display: flex; align-items: center; gap: 12px; min-width: 0; }
.logo { width: 42px; height: 42px; flex: none; display: grid; place-items: center; border-radius: 13px; background: var(--grad); border: 1px solid var(--border); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
.logo svg { width: 26px; height: 26px; }
.eyebrow { color: var(--accent); font-size: .66rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
.title { margin: 1px 0 2px; color: var(--text); font-size: 1.4rem; line-height: 1.12; letter-spacing: -.03em; font-weight: 800; }
.subtitle { color: var(--muted); font-size: .84rem; line-height: 1.4; }
.status-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pill { box-sizing: border-box; display: inline-flex; align-items: center; gap: 8px; min-height: 32px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 999px; color: var(--text); background: var(--surface-2); font-size: .66rem; line-height: 1; font-weight: 700; letter-spacing: .05em; white-space: nowrap; }
.dot { display: inline-block; flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--trust); }
.dot.loading { background: var(--verify); animation: dot-pulse 1.5s ease-in-out infinite; }
.pill .dot.loading { box-sizing: border-box; width: 8px; height: 8px; margin: 0; padding: 0; border: 0; }
.dot.off { background: var(--error); }

.st-key-input_card { padding: 14px 16px 12px; border: 1px solid var(--border); border-radius: 18px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
.ask-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.ask-head b { color: var(--text); font-size: 1rem; font-weight: 750; letter-spacing: -.02em; }
.ask-head span { color: var(--muted); font-size: .81rem; }
.card-head { color: var(--text); font-size: 1.05rem; font-weight: 750; }
.card-sub { color: var(--muted); font-size: .88rem; }
.mini-label { margin: 8px 0 4px; color: var(--muted); font-size: .66rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
div[data-testid="stTextInput"] input, .stTextInput input { box-sizing: border-box; min-height: 3rem; padding-inline: 14px; color: var(--text) !important; caret-color: var(--accent); background: var(--surface-2) !important; border-radius: 12px; font-size: .98rem; line-height: 1.4; }
div[data-testid="stTextInput"] input::placeholder, .stTextInput input::placeholder { color: var(--muted) !important; opacity: 1; }
div[data-testid="stTextInput"] [data-baseweb="input"], .stTextInput [data-baseweb="input"] { min-height: 3rem; border: 1px solid var(--border); border-radius: 12px; background: var(--surface-2); box-shadow: none; transition: border-color .16s ease, box-shadow .16s ease; }
div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within, .stTextInput [data-baseweb="input"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px var(--glow-1); }
.st-key-input_card [data-testid="stForm"] { border: 0; padding: 0; }

[data-testid="stButton"] button, .stButton > button, [data-testid="stFormSubmitButton"] button, .stFormSubmitButton > button { box-sizing: border-box; min-height: 2.75rem; border: 1px solid var(--border); border-radius: 11px; color: var(--text); background: var(--surface-2); box-shadow: none; font-weight: 650; transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease, background-color .16s ease; }
[data-testid="stButton"] button:hover, .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover, .stFormSubmitButton > button:hover { transform: translateY(-1px); border-color: var(--accent); background: var(--surface-2); color: var(--text); box-shadow: 0 5px 16px var(--glow-1); }
[data-testid="stButton"] button:focus-visible, .stButton > button:focus-visible, [data-testid="stFormSubmitButton"] button:focus-visible, .stFormSubmitButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
[data-testid="stButton"] button p, .stButton > button p, [data-testid="stFormSubmitButton"] button p, .stFormSubmitButton > button p { color: inherit; font-size: .82rem; line-height: 1.3; font-weight: inherit; white-space: normal; }
[data-testid="stFormSubmitButton"] button, .stFormSubmitButton > button { width: 100%; min-height: 3rem; }
[data-testid="stFormSubmitButton"] button[kind="primary"], .stFormSubmitButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"] { border: 0; color: var(--on-accent); background: var(--grad); font-weight: 800; box-shadow: 0 7px 22px var(--glow-1); }
[data-testid="stFormSubmitButton"] button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p { color: var(--on-accent); font-size: .92rem; font-weight: 800; }
[data-testid="stFormSubmitButton"] button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover { transform: translateY(-1px); color: var(--on-accent); box-shadow: 0 10px 26px var(--glow-1); }
[data-testid="stButton"] button:disabled, [data-testid="stFormSubmitButton"] button:disabled { cursor: not-allowed; opacity: .58; transform: none; box-shadow: none; }
.st-key-input_card .stButton > button { min-height: 2rem; padding: .2rem .55rem; border-radius: 999px; }
.st-key-input_card .stButton > button p { font-size: .72rem; }
.st-key-clear_button button { color: var(--muted); background: transparent; border-style: dashed; }

.card { border: 1px solid var(--border); border-radius: 17px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
.section-title { margin: 26px 0 3px; color: var(--text); font-size: 1.12rem; font-weight: 750; }
.section-sub { color: var(--muted); font-size: .88rem; line-height: 1.45; margin-bottom: 14px; }
.notice { display: flex; align-items: flex-start; gap: 10px; margin-top: 12px; padding: 12px 14px; border: 1px solid var(--border); border-radius: 13px; color: var(--text); background: var(--surface); animation: rise .22s ease both; }
.notice b { display: block; margin-bottom: 2px; }
.notice span { font-size: .9rem; line-height: 1.45; }
.notice.info { border-color: var(--info-bd); color: var(--text); background: var(--info-bg); }
.notice.warn { border-color: var(--verify-bd); color: var(--text); background: var(--verify-bg); }
.notice.error { border-color: var(--error-bd); color: var(--text); background: var(--error-bg); }

/* Answer card and final decision */
.rc-wrap { margin: 14px 0 30px; padding: 1px; border: 1px solid var(--border); border-radius: 18px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); animation: rise .3s ease both; }
.rc-wrap.trust { border-color: var(--trust-bd); }
.rc-wrap.verify { border-color: var(--verify-bd); }
.rc { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(230px, .75fr); gap: 22px; padding: 22px; border-radius: 17px; background: var(--surface); }
.answer { margin: 6px 0 16px; color: var(--text); font-size: 1.85rem; line-height: 1.25; font-weight: 760; letter-spacing: -.025em; overflow-wrap: anywhere; }
.answer-word { display: inline-block; }
.badge-row { display: flex; align-items: center; flex-wrap: wrap; gap: 9px; margin: 12px 0; }
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px; border: 1px solid var(--border); border-radius: 999px; font-size: .86rem; font-weight: 800; letter-spacing: .06em; cursor: default; user-select: none; box-shadow: none; }
.badge.trust { color: var(--trust); background: var(--trust-bg); border-color: var(--trust-bd); }
.badge.verify { color: var(--verify); background: var(--verify-bg); border-color: var(--verify-bd); }
.badge svg { width: 20px; height: 20px; flex: none; }
.verify-link { display: inline-flex; align-items: center; min-height: 34px; padding: 6px 11px; border: 1px solid var(--info-bd); border-radius: 999px; color: var(--accent); background: var(--info-bg); font-size: .78rem; font-weight: 700; text-decoration: none; }
.verify-link:hover { border-color: var(--accent); color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }
.thr-chip { padding: 7px 10px; border: 1px solid var(--border); border-radius: 10px; color: var(--muted); background: var(--surface-2); font-size: .78rem; font-weight: 650; }
.decision-msg { margin: 10px 0 12px; color: var(--text); font-size: .92rem; line-height: 1.55; }
.meta-line { margin: 0 0 10px; color: var(--muted); font-size: .76rem; }
.qecho { margin: 0 0 12px; color: var(--muted); font-size: .84rem; overflow-wrap: anywhere; }
.qecho b { color: var(--text); }
.disclaimer { padding-top: 10px; border-top: 1px dashed var(--border); color: var(--muted); font-size: .76rem; line-height: 1.5; }
.gauge-col { display: flex; flex-direction: column; align-items: center; justify-content: center; min-width: 0; padding: 16px; border: 1px solid var(--border); border-radius: 15px; background: var(--surface-2); }
.gauge-col .eyebrow { margin-bottom: 6px; color: var(--muted); }
.gauge { display: block; width: 100%; max-width: 250px; filter: drop-shadow(0 10px 18px var(--border)); }
.ring-prog { stroke-dashoffset: var(--off); }
.meter { position: relative; width: 100%; max-width: 320px; height: 9px; margin: 18px 0 0; border-radius: 99px; background: var(--track); }
.meter-fill { height: 100%; width: var(--w); border-radius: 99px; }
.meter-fill.trust { background: linear-gradient(90deg, var(--trust), var(--accent-2)); }
.meter-fill.verify { background: linear-gradient(90deg, var(--verify), var(--error)); }
.meter-mark { position: absolute; top: -6px; width: 2px; height: 22px; border-radius: 3px; background: var(--text); transform: translateX(-1px); }
.meter-mark span { position: absolute; top: 27px; left: 50%; transform: translateX(-50%); color: var(--muted); font-size: .68rem; font-weight: 700; white-space: nowrap; }
.meter-cap { width: 100%; min-height: 2.8em; margin-top: 42px; padding: 0 8px; color: var(--text); font-size: .82rem; font-weight: 700; line-height: 1.5; text-align: center; overflow-wrap: anywhere; }
.meter-cap:empty { display: none; }
.meter-cap em { color: var(--text); font-style: normal; }
.meter-cap .trust { color: var(--trust); }
.meter-cap .verify { color: var(--verify); }
.verification-note { margin: 2px 0 12px; padding: 11px 13px; border: 1px solid var(--info-bd); border-radius: 12px; color: var(--text); background: var(--info-bg); font-size: .84rem; line-height: 1.5; }
.verification-note b { display: block; margin-bottom: 3px; }
.verification-note.verified { border-color: var(--trust-bd); background: var(--trust-bg); }
.verification-note.failed { border-color: var(--verify-bd); background: var(--verify-bg); }
.verification-note a { color: var(--accent); font-weight: 700; text-underline-offset: 2px; }
.quality-raw { margin: 2px 0 12px; padding: 11px 13px; border: 1px solid var(--error-bd); border-radius: 12px; color: var(--text); background: var(--error-bg); font-size: .84rem; line-height: 1.5; }
.quality-raw b { display: block; margin-top: 3px; overflow-wrap: anywhere; }
.quality-label { display: inline-flex; margin-bottom: 7px; padding: 4px 8px; border-radius: 99px; color: var(--error); background: var(--error-bg); font-size: .66rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
.hint-note { margin: 2px 0 12px; padding: 10px 13px; border: 1px solid var(--info-bd); border-radius: 12px; color: var(--text); background: var(--info-bg); font-size: .82rem; line-height: 1.5; }
.hint-note b { display: block; margin-bottom: 2px; }
.fresh .answer-word { animation: wordin .35s ease both; animation-delay: calc(var(--i) * 45ms); }
.fresh .badge-row { animation: badgein .35s ease both; animation-delay: var(--answer-delay, 0ms); }
.fresh .ring-prog.fresh-ring { animation: ringfill 1.5s cubic-bezier(.2,.8,.2,1) both; }

/* Reliability signals, samples and technical details */
.grid5 { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.mcard { --metric-accent: var(--accent); position: relative; overflow: hidden; padding: 15px; border: 1px solid var(--border); border-radius: 15px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
.mcard::before { position: absolute; inset: 0 0 auto; height: 3px; background: linear-gradient(90deg, var(--metric-accent), transparent); content: ""; }
.mlabel { min-height: 2em; color: var(--muted); font-size: .68rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; }
.mvalue { margin: 5px 0 7px; color: var(--text); font-size: 1.7rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.mbar { height: 6px; margin-bottom: 9px; overflow: hidden; border-radius: 99px; background: var(--surface-2); }
.mbar i { display: block; height: 100%; width: var(--w); border-radius: inherit; background: linear-gradient(90deg, var(--metric-accent), var(--accent-2)); }
.msub { margin-bottom: 5px; color: var(--text); font-size: .78rem; font-weight: 650; }
.mnote { color: var(--muted); font-size: .76rem; line-height: 1.45; }
.s-banner { margin-bottom: 12px; padding: 11px 14px; border: 1px solid var(--info-bd); border-radius: 12px; color: var(--text); background: var(--info-bg); font-size: .88rem; }
.s-banner.split { border-color: var(--verify-bd); background: var(--verify-bg); }
.sgrid { display: grid; gap: 10px; }
.scard { display: flex; align-items: flex-start; gap: 12px; min-width: 0; padding: 12px 14px; border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 12px; background: var(--surface); }
.scard.diff { border-left-color: var(--verify); }
.snum { color: var(--accent); font-size: 1.15rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.scard.diff .snum { color: var(--verify); }
.sbody { flex: 1; min-width: 0; }
.stext { color: var(--text); font-size: .95rem; font-weight: 600; overflow-wrap: anywhere; }
.skey { margin-top: 4px; color: var(--muted); font: .74rem ui-monospace, "Cascadia Mono", Consolas, monospace; overflow-wrap: anywhere; }
.tag { flex: none; padding: 4px 8px; border-radius: 99px; font-size: .65rem; font-weight: 700; text-transform: uppercase; }
.tag.same { color: var(--trust); background: var(--trust-bg); }
.tag.diff { color: var(--verify); background: var(--verify-bg); }
.s-foot { margin-top: 10px; color: var(--muted); font-size: .76rem; }
.kvgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; }
.kv { padding: 11px 14px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface-2); }
.kv small { display: block; margin-bottom: 3px; color: var(--muted); font-size: .65rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.kv b { color: var(--text); font-size: .91rem; overflow-wrap: anywhere; }
.feat-title { margin: 18px 0 8px; color: var(--muted); font-size: .78rem; font-weight: 700; }
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.chip { padding: 5px 9px; border: 1px solid var(--border); border-radius: 99px; color: var(--muted); background: var(--surface-2); font: .72rem ui-monospace, "Cascadia Mono", Consolas, monospace; }

/* Native Streamlit surfaces */
[data-testid="stExpander"] { overflow: hidden; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 14px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
[data-testid="stExpander"] summary { padding: .8rem 1rem; color: var(--text); font-weight: 700; }
[data-testid="stExpander"] summary p { color: var(--text); font-size: .94rem; font-weight: 700; }
[data-testid="stExpanderDetails"], [data-testid="stExpander"] details { color: var(--text); background: var(--surface); }
[data-testid="stExpander"] summary svg { color: var(--muted); fill: currentColor; }
[data-testid="stCode"], [data-testid="stCode"] pre, code, pre { color: var(--text); background: var(--code-bg); }
[data-testid="stMarkdownContainer"], .stMarkdown { color: var(--text); }
a { color: var(--accent); }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { color: var(--text); background: var(--surface); border-color: var(--border); }
[data-baseweb="popover"] *, [data-baseweb="menu"] *, [role="listbox"] * { color: inherit; }
[role="option"]:hover, [role="option"][aria-selected="true"] { background: var(--surface-2); }
.st-key-theme_choice [data-testid="stSegmentedControl"] { justify-content: flex-end; }
.st-key-theme_choice [data-testid="stSegmentedControl"] button, .st-key-theme_choice [data-testid="stSegmentedControl"] label { color: var(--muted); background: var(--surface-2); border-color: var(--border); }
.st-key-theme_choice [data-testid="stSegmentedControl"] [aria-pressed="true"], .st-key-theme_choice [data-testid="stSegmentedControl"] [aria-checked="true"] { color: var(--on-accent); background: var(--grad); }
.st-key-theme_choice [data-testid="stRadio"] label { color: var(--text); }
[data-testid="stSidebar"] { border-right: 1px solid var(--border); background: var(--surface); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: var(--surface); }
[data-testid="stSidebar"] * { color: inherit; }
.side-text { margin: 8px 0 14px; color: var(--muted); font-size: .88rem; line-height: 1.5; }
.side-kv { display: flex; justify-content: space-between; gap: 10px; padding: 9px 0; border-bottom: 1px solid var(--border); color: var(--text); font-size: .84rem; }
.side-kv span, .policy small { color: var(--muted); }
.side-kv b { text-align: right; }
.policy { margin: 14px 0; padding: 12px; border: 1px solid var(--border); border-radius: 13px; background: var(--surface-2); font-size: .83rem; line-height: 1.8; }
.policy .t { color: var(--trust); font-weight: 800; }
.policy .v { color: var(--verify); font-weight: 800; }
.policy small { display: block; margin-bottom: 4px; font-size: .64rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.side-note { padding: 11px 12px; border: 1px solid var(--verify-bd); border-radius: 12px; color: var(--text) !important; background: var(--verify-bg); font-size: .79rem; line-height: 1.5; }
.hist { margin-top: 18px; }
.hist-item { display: flex; align-items: center; gap: 9px; padding: 7px 0; border-bottom: 1px solid var(--border); color: var(--text); font-size: .8rem; }
.hist-item i { flex: none; width: 8px; height: 8px; border-radius: 50%; }
.hist-item i.trust { background: var(--trust); }
.hist-item i.verify { background: var(--verify); }
.hist-item em { flex: 1; min-width: 0; overflow: hidden; font-style: normal; text-overflow: ellipsis; white-space: nowrap; }
.hist-item b { flex: none; font-variant-numeric: tabular-nums; }
.foot { margin-top: 20px; color: var(--muted); font-size: .74rem; text-align: center; }

/* Live answer preparation card */
.loading { position: relative; display: block; overflow: hidden; margin-top: 12px; padding: 13px; border: 1px solid var(--border); border-radius: 18px; background: var(--surface); box-shadow: var(--shadow), inset 0 1px 0 var(--hairline); }
.answer-skeleton { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(200px, .75fr); align-items: center; gap: 22px; padding: 18px 20px; border: 1px solid var(--border); border-radius: 15px; background: var(--surface); box-shadow: inset 0 1px 0 var(--hairline); }
.skeleton-heading { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.skeleton-heading .eyebrow { color: var(--accent); }
.preparing-tag { padding: 4px 8px; border: 1px solid var(--info-bd); border-radius: 99px; color: var(--accent); background: var(--info-bg); font-size: .62rem; font-weight: 800; letter-spacing: .08em; }
.skeleton-lines { display: grid; gap: 10px; }
.skeleton-line { height: 13px; border-radius: 99px; background: linear-gradient(100deg, var(--surface-2) 20%, var(--border) 45%, var(--surface-2) 70%); background-size: 220% 100%; animation: shimmer 1.8s ease-in-out infinite; }
.skeleton-line:nth-child(1) { width: 92%; }
.skeleton-line:nth-child(2) { width: 76%; }
.skeleton-line:nth-child(3) { width: 54%; }
.thinking { display: inline-flex; align-items: center; gap: 5px; margin-top: 17px; color: var(--muted); font-size: .77rem; }
.thinking i { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: think 1.2s ease-in-out infinite; }
.thinking i:nth-child(2) { animation-delay: .15s; }
.thinking i:nth-child(3) { animation-delay: .3s; }
.skeleton-gauge { position: relative; display: grid; place-items: center; width: min(100%, 200px); aspect-ratio: 1; justify-self: center; border: 1px solid var(--border); border-radius: 50%; background: var(--surface-2); }
.skeleton-gauge::before { position: absolute; inset: 14px; border: 4px solid var(--border); border-top-color: var(--accent); border-right-color: var(--accent-2); border-radius: 50%; animation: rotate 4s linear infinite; content: ""; }
.skeleton-gauge span { position: relative; color: var(--muted); font-size: .68rem; font-weight: 800; letter-spacing: .15em; }
.lmain { padding: 13px 8px 2px; }
.lhead { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.ltitle { color: var(--text); font-size: .98rem; font-weight: 750; }
.lclock { flex: none; padding: 3px 9px; border: 1px solid var(--border); border-radius: 99px; color: var(--muted); background: var(--surface-2); font-size: .72rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.lsub { margin: 3px 0 8px; color: var(--muted); font-size: .82rem; }
.pipe { display: grid; gap: 6px; margin: 10px 0 0; padding: 0; list-style: none; }
.pipe li { display: flex; align-items: center; gap: 9px; min-width: 0; color: var(--muted); font-size: .82rem; }
.pipe li > span:nth-child(2) { flex: 1; min-width: 0; }
.pipe li small { margin-left: 6px; color: var(--muted); font-size: .7rem; font-weight: 600; }
.pipe .pd { box-sizing: border-box; display: grid; flex: none; place-items: center; width: 19px; height: 19px; border: 2px solid var(--border); border-radius: 50%; color: var(--on-accent); font-size: .62rem; font-weight: 800; }
.pipe li.done { color: var(--text); }
.pipe li.done .pd { border-color: var(--trust); color: var(--on-accent); background: var(--trust); }
.pipe li.active { color: var(--text); font-weight: 700; }
.pipe li.active small { color: var(--accent); }
.pipe li.active .pd { border-color: var(--accent); background: var(--glow-1); }
.pips { display: inline-flex; flex: none; gap: 4px; }
.pip { display: block; width: 8px; height: 8px; border-radius: 50%; background: var(--border); }
.pip.done { background: var(--trust); }
.pip.active { background: var(--accent); box-shadow: 0 0 0 3px var(--glow-1); }
.candidate-label, .pv-label { margin-top: 11px; color: var(--muted); font-size: .62rem; font-weight: 750; letter-spacing: .11em; text-transform: uppercase; }
.candidates { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.candidate { max-width: 100%; padding: 5px 9px; overflow: hidden; border: 1px solid var(--border); border-radius: 999px; color: var(--text); background: var(--surface-2); font-size: .75rem; text-overflow: ellipsis; white-space: nowrap; }
.candidate.new { animation: pop .25s ease both; }
.pvs { display: grid; gap: 5px; margin-top: 5px; }
.pv { display: flex; gap: 9px; min-width: 0; padding: 6px 9px; border: 1px solid var(--border); border-radius: 9px; color: var(--text); background: var(--surface-2); font-size: .78rem; }
.pv.new { animation: pop .25s ease both; }
.pv b { flex: none; color: var(--accent); font-variant-numeric: tabular-nums; }
.pv span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lbar { height: 3px; margin: 10px -13px -13px; background: var(--surface-2); }
.lbar::after { display: block; width: 38%; height: 100%; border-radius: 99px; background: var(--grad); animation: shimmer 1.4s linear infinite; content: ""; }

.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 13px; margin-top: 22px; }
.step { padding: 17px; border-radius: 15px; }
.step-num { display: grid; place-items: center; width: 30px; height: 30px; margin-bottom: 9px; border-radius: 10px; color: var(--on-accent); background: var(--grad); font-weight: 800; }
.step-title { color: var(--text); margin-bottom: 3px; font-weight: 750; }
.step-text { color: var(--muted); font-size: .85rem; line-height: 1.5; }

@media (max-width: 860px) {
    .rc, .answer-skeleton { grid-template-columns: minmax(0, 1fr); }
    .gauge-col { border-top: 1px solid var(--border); }
    .skeleton-gauge { width: min(100%, 180px); }
}
@media (max-width: 640px) {
    .block-container { padding: .7rem .8rem 1.5rem; }
    .hero { align-items: flex-start; gap: 9px; padding-bottom: 10px; }
    .brand { align-items: flex-start; gap: 9px; }
    .logo { width: 38px; height: 38px; border-radius: 11px; }
    .logo svg { width: 24px; height: 24px; }
    .title { font-size: 1.18rem; }
    .subtitle { font-size: .77rem; }
    .status-row { width: 100%; gap: 6px; }
    .pill { min-height: 30px; padding: 5px 8px; font-size: .6rem; }
    .st-key-input_card { padding: 12px 12px 10px; border-radius: 15px; }
    .ask-head { gap: 4px 8px; }
    .ask-head span { font-size: .75rem; }
    div[data-testid="stTextInput"] input, .stTextInput input, div[data-testid="stTextInput"] [data-baseweb="input"], .stTextInput [data-baseweb="input"] { min-height: 2.8rem; }
    [data-testid="stButton"] button p, .stButton > button p { font-size: .7rem; }
    .st-key-example_chips [data-testid="stHorizontalBlock"] { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .st-key-example_chips [data-testid="column"] { width: auto !important; min-width: 0 !important; flex: initial !important; }
    .st-key-example_chips button { width: 100%; }
    .rc { gap: 14px; padding: 15px 13px; }
    .answer { font-size: 1.48rem; }
    .gauge-col { padding: 12px; }
    .meter-cap { margin-top: 42px; }
    .answer-skeleton { gap: 14px; padding: 14px 12px; }
    .skeleton-gauge { width: min(100%, 150px); }
    .lmain { padding: 11px 3px 1px; }
    .pipe li { font-size: .75rem; }
    .pipe li small { font-size: .66rem; }
    .steps { grid-template-columns: minmax(0,1fr); gap: 9px; }
    .step { padding: 14px; }
    .grid5 { grid-template-columns: repeat(auto-fit, minmax(138px, 1fr)); gap: 8px; }
    .section-title { margin-top: 22px; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { scroll-behavior: auto !important; animation: none !important; transition: none !important; }
}
</style>
"""

ORBS = '<div class="orbs"><div class="orb a"></div><div class="orb b"></div><div class="orb c"></div></div>'


# ============================================================
# SMALL HELPERS
# ============================================================

def html_block(markup: str) -> str:
    """Strip indentation and blank lines so Markdown keeps the HTML intact."""
    return "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )


def render(markup: str) -> None:
    st.markdown(html_block(markup), unsafe_allow_html=True)


def esc(value: object) -> str:
    """HTML-escape model/user text. '$' is escaped so Markdown math never triggers."""
    return html.escape(str(value), quote=True).replace("$", "&#36;")


def pct_plain(value: float) -> str:
    """0.82 -> '82', 0.825 -> '82.5'."""
    return f"{value * 100:.1f}".rstrip("0").rstrip(".")


def token_probability(log_prob: float) -> float:
    return math.exp(min(log_prob, 0.0))


def is_open_ended(question: str) -> bool:
    """Heuristic UI hint only. It never changes a decision; it just explains N/A."""
    pattern = (
        r"\b(explain|describe|discuss|summari[sz]e|write|essay|story|poem|compare|"
        r"overview|tell me about|pros and cons)\b|\bin\s+\d+\s+words\b|\bhow\s+(?:does|do)\b"
    )
    return bool(re.search(pattern, str(question or ""), re.IGNORECASE))


def stage_step(stage: str) -> int:
    """Map a pipeline stage title to the index of the visible progress step."""
    for prefix, index in (
        ("Loading", 0),
        ("Sampling", 1),
        ("Extracting", 2),
        ("Classifying", 2),
        ("Running", 3),
        ("Finalizing", 4),
    ):
        if stage.startswith(prefix):
            return index
    return 1


# ============================================================
# MODEL LOADING (once per process)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_gate():
    # Imported here, not at module level: this pulls in torch + transformers.
    from src.gate.reliability_gate import ReliabilityGate

    return ReliabilityGate()


def try_load_gate() -> tuple[ReliabilityGate | None, dict | None]:
    """Load the gate; on failure return a friendly message instead of a traceback."""
    try:
        return load_gate(), None
    except FileNotFoundError as exc:
        logger.exception("Reliability model file missing")
        return None, {
            "title": "Reliability model not found",
            "body": (
                "The trained gate file could not be found. Generate a compatible "
                "V5 artifact through the supported benchmark and calibration workflow."
            ),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    except ValueError as exc:
        logger.exception("Reliability model file invalid")
        return None, {
            "title": "Reliability model is invalid",
            "body": (
                "The gate file was found but is not valid or does not match "
                "the expected 15 reliability features."
            ),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        logger.exception("Local model failed to load")
        return None, {
            "title": "Local model could not be loaded",
            "body": (
                "Qwen/Qwen3-0.6B or its runtime failed to start. Check that "
                "the model is available locally and that your CUDA setup works."
            ),
            "detail": f"{type(exc).__name__}: {exc}",
        }


def runtime_label(gate: ReliabilityGate | None) -> str:
    """Describe the device the local LLM is actually running on."""
    device = getattr(getattr(gate, "llm", None), "device", None)

    if device is None:
        return "Unknown"

    device_type = getattr(device, "type", str(device))

    if device_type == "cuda":
        name = ""
        if torch is not None:
            try:
                name = torch.cuda.get_device_name(device)
            except Exception:
                name = ""
        return f"CUDA · {name}" if name else "CUDA"

    return "CPU"


# ============================================================
# INFERENCE + RESULT VALIDATION
# ============================================================

@st.cache_resource(show_spinner=False)
def process_state() -> dict:
    """Status shared by every browser session of this server process."""
    return {
        "loaded": False,
        "error": None,
        "runtime": None,
        "threshold": None,
        # The active artifact is not considered calibrated until ReliabilityGate loads it
        # and validates its fingerprint. File existence alone is not sufficient.
        "calibrated": False,
        "calibration_note": (
            "Calibration has not yet been validated for the active configuration. The gate checks "
            "the artifact fingerprint when loaded; until then, every answer is VERIFY and no "
            "reliability score is reported."
        ),
        "lock": threading.Lock(),
    }


def ensure_gate():
    """Load (once per process, cached) and record real status. Called on first click only."""
    global torch
    state = process_state()
    gate, err = try_load_gate()
    if gate is None:
        state.update(loaded=False, error=err)
        return None, err
    try:
        import torch as _torch

        torch = _torch
    except Exception:
        pass
    state.update(
        loaded=True,
        error=None,
        runtime=runtime_label(gate),
        threshold=gate.threshold,
        calibrated=bool(gate.calibrated),
        calibration_note=gate.calibration_note,
    )
    return gate, None


def answer_groups(question: str, answers: list[str]) -> tuple[list[str], str, int]:
    """Group samples with the gate's conservative equivalence rule."""
    keys: list[str] = []

    for answer in answers:
        try:
            key = extract_answer_key(question, answer)
        except Exception:
            key = " ".join(answer.lower().split())
        keys.append(key)

    if not keys:
        return keys, "", 0

    representatives: list[tuple[str, str]] = []
    grouped_keys: list[str] = []
    for answer, key in zip(answers, keys):
        for representative_answer, representative_key in representatives:
            if answers_are_equivalent(question, answer, representative_answer, key, representative_key):
                grouped_keys.append(representative_key)
                break
        else:
            representatives.append((answer, key))
            grouped_keys.append(key)

    counts = Counter(grouped_keys)
    return grouped_keys, counts.most_common(1)[0][0], len(counts)


def build_view(result: object, latency: float, runtime: str) -> dict:
    """Validate the GateResult and convert it into a plain dict for rendering."""
    probability = float(result.probability_correct)
    threshold = float(result.threshold)

    if not (math.isfinite(probability) and math.isfinite(threshold)):
        raise ValueError("Non-finite probability or threshold.")

    probability = min(max(probability, 0.0), 1.0)

    model_decision = str(result.decision).strip().upper()
    if model_decision not in {"TRUST", "VERIFY"}:
        # An invalid producer decision is an integration failure, not permission
        # to reconstruct TRUST from a score in the UI.
        model_decision = "VERIFY"

    samples = [str(answer) for answer in (result.raw_answers or [])]
    question = str(result.question)
    answer = str(result.answer or "").strip()
    quality = assess_response_quality(question, answer)
    # Fail closed if any trust-critical field is missing. A useful-looking answer is
    # not independently verified merely because the model samples agree.
    calibrated = bool(getattr(result, "calibrated", False))
    reliability_reported = bool(getattr(result, "reliability_reported", False))
    verification_status = str(getattr(result, "fact_verification_status", "UNVERIFIED")).upper()
    verification_reason = str(getattr(result, "fact_verification_reason", "No independent verification result."))
    verification_evidence = str(getattr(result, "fact_verification_evidence", ""))
    verification_source_url = str(getattr(result, "fact_verification_source_url", ""))
    verification_supported = bool(getattr(result, "fact_verification_supported", False))
    material_disagreement = bool(getattr(result, "material_disagreement", False))

    decision = (
        "VERIFY"
        if (not quality.is_useful or not calibrated or not reliability_reported
            or material_disagreement or verification_status != "VERIFIED"
            or not verification_supported)
        else model_decision
    )
    score_valid = bool(
        quality.is_useful and calibrated and reliability_reported
        and not material_disagreement
        and verification_status == "VERIFIED" and verification_supported
    )
    keys, winner, distinct = answer_groups(question, samples)

    view = {
        "question": question,
        "answer": answer,
        "probability": probability,
        "threshold": threshold,
        "decision": decision,
        "model_decision": model_decision,
        "quality_ok": quality.is_useful,
        "quality_reason": quality.reason,
        "calibrated": calibrated,
        "score_valid": score_valid,
        "fact_verification_status": verification_status,
        "fact_verification_reason": verification_reason,
        "fact_verification_evidence": verification_evidence,
        "fact_verification_source_url": verification_source_url,
        "fact_verification_supported": verification_supported,
        "material_disagreement": material_disagreement,
        "agreement": float(result.answer_agreement),
        "margin": float(result.consensus_margin),
        "entropy": float(result.answer_key_entropy),
        "mean_lp": float(result.mean_logprob_avg),
        "min_lp": float(result.min_logprob_avg),
        "samples": samples,
        "keys": keys,
        "winner_key": winner,
        "distinct": distinct,
        "latency": latency,
        "runtime": runtime,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }

    numeric = ("agreement", "margin", "entropy", "mean_lp", "min_lp")
    if not all(math.isfinite(view[name]) for name in numeric):
        raise ValueError("Non-finite reliability signal.")

    return view


def run_check(gate: ReliabilityGate, question: str) -> dict:
    started = perf_counter()
    result = gate.predict(question=question, return_all_answers=True)
    latency = perf_counter() - started
    return build_view(result, latency, runtime_label(gate))


def run_with_stage_hooks(gate: ReliabilityGate, question: str, on_stage, on_progress=None) -> dict:
    """Report actual gate call boundaries while keeping instrumentation in the UI layer.

    on_stage(title) fires at each pipeline stage. on_progress(done, total, answer_text) fires
    after every individual sample so the interface can show answers as they arrive. Each
    callback is also a Streamlit call, which is what lets a Clear/new question interrupt a run
    at the next sample boundary instead of after all five samples.
    """
    from src.gate import reliability_gate as gate_module

    model = gate.model
    original_sample = gate_module.sample_answers
    original_verify = gate_module.verify_answer
    original_generate = getattr(gate_module, "generate_one", None)
    original_predict_proba = model.predict_proba
    original_build_features = gate._build_features
    original_consensus = gate._select_consensus
    counter = {"done": 0}

    def tracked_sampling(*args, **kwargs):
        counter["done"] = 0
        on_stage(f"Sampling {N_SAMPLES} answers")
        return original_sample(*args, **kwargs)

    def tracked_generate(*args, **kwargs):
        sample = original_generate(*args, **kwargs)
        counter["done"] += 1
        if on_progress is not None:
            try:
                on_progress(counter["done"], N_SAMPLES, str(getattr(sample, "answer", "")))
            except Exception:  # a display problem must never break inference
                logger.exception("Progress callback failed")
        return sample

    def tracked_features(*args, **kwargs):
        on_stage("Extracting reliability features")
        return original_build_features(*args, **kwargs)

    def tracked_classification(*args, **kwargs):
        on_stage("Classifying answer reliability")
        return original_predict_proba(*args, **kwargs)

    def tracked_consensus(*args, **kwargs):
        result = original_consensus(*args, **kwargs)
        if not result[1].get("quality_ok"):
            on_stage("Finalizing the recommendation")
        return result

    def tracked_verification(*args, **kwargs):
        on_stage("Running independent verification")
        try:
            return original_verify(*args, **kwargs)
        finally:
            on_stage("Finalizing the recommendation")

    patches = []

    def install_patch(target, name, replacement):
        had_instance_value = name in vars(target)
        previous_value = getattr(target, name)
        patches.append((target, name, had_instance_value, previous_value))
        setattr(target, name, replacement)

    try:
        install_patch(gate_module, "sample_answers", tracked_sampling)
        install_patch(gate_module, "verify_answer", tracked_verification)
        if original_generate is not None:
            install_patch(gate_module, "generate_one", tracked_generate)
        install_patch(model, "predict_proba", tracked_classification)
        install_patch(gate, "_build_features", tracked_features)
        install_patch(gate, "_select_consensus", tracked_consensus)
        return run_check(gate, question)
    finally:
        for target, name, had_instance_value, previous_value in reversed(patches):
            if had_instance_value:
                setattr(target, name, previous_value)
            else:
                delattr(target, name)


def load_and_run(gate_was_loaded: bool, question: str, on_stage, on_progress=None):
    if not gate_was_loaded:
        on_stage(f"Loading local model: {MODEL_NAME.split('/')[-1]}")
    gate, error = ensure_gate()
    if gate is None:
        return None, error, None
    return gate, None, run_with_stage_hooks(gate, question, on_stage, on_progress)


def acquire_run_lock(state: dict, on_wait, timeout: float = 240.0) -> bool:
    """Wait (instead of failing) while another session's check finishes."""
    started = perf_counter()
    last_paint = -1.0
    while True:
        if state["lock"].acquire(timeout=0.3):
            return True
        waited = perf_counter() - started
        if waited >= timeout:
            return False
        if last_paint < 0 or waited - last_paint >= 1.0:
            last_paint = waited
            on_wait(waited)


# ============================================================
# HTML BUILDERS
# ============================================================

def hero_html(status: str, runtime: str | None = None) -> str:
    labels = {
        "online": ("LOCAL MODEL • ONLINE", ""),
        "loading": ("LOCAL MODEL • LOADING", "loading"),
        "idle": ("MODEL LOADS ON FIRST QUESTION", "loading"),
        "offline": ("LOCAL MODEL • OFFLINE", "off"),
    }
    label, dot_class = labels[status]

    runtime_pill = (
        f'<div class="pill">{esc(runtime)}</div>' if runtime else ""
    )

    return f"""
    <div class="hero">
        <div class="brand">
            <div class="logo">
                <svg width="34" height="34" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <defs>
                        <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0" stop-color="var(--accent)"/>
                            <stop offset="1" stop-color="var(--accent-2)"/>
                        </linearGradient>
                    </defs>
                    <path d="M12 2.5l8 3v6.2c0 4.6-3.2 8.3-8 9.8-4.8-1.5-8-5.2-8-9.8V5.5l8-3z" fill="url(#lg)"/>
                    <path d="M8.2 12.2l2.6 2.6 5-5.4" stroke="var(--on-accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div>
                <div class="eyebrow">AI Reliability Platform</div>
                <div class="title">Local LLM Reliability Gate</div>
                <div class="subtitle">Know when to trust a local model — and when to verify.</div>
            </div>
        </div>
        <div class="status-row">
            <div class="pill"><span class="dot {dot_class}"></span>{label}</div>
            <div class="pill">{esc(MODEL_NAME.split("/")[-1])}</div>
            {runtime_pill}
        </div>
    </div>
    """


def notice_html(kind: str, title: str, body: str) -> str:
    icon = {"info": "ℹ️", "warn": "⚠️", "error": "⛔"}.get(kind, "ℹ️")
    return f"""
    <div class="notice {kind}">
        <div>{icon}</div>
        <div><b>{esc(title)}</b><span>{esc(body)}</span></div>
    </div>
    """


def calibration_banner_html(loaded: bool, calibrated: bool, note: str) -> str:
    if not loaded:
        return notice_html(
            "info",
            "Model not loaded yet",
            "The local model loads automatically when you check your first question.",
        )
    if not calibrated:
        return notice_html("warn", "Calibration refresh required", note)
    return ""


def loading_html(
    title: str,
    detail: str,
    active: int | None = None,
    samples_done: int = 0,
    samples_total: int = 0,
    previews: list | tuple = (),
    elapsed: float = 0.0,
    model_ready: bool = False,
) -> str:
    """Show a result-shaped preparation card and the live inference pipeline."""
    steps = (
        ("Load model", "waking up the local Qwen model"),
        ("Sample answers", "writing independent answers"),
        ("Measure agreement", "comparing answers and token confidence"),
        ("Verify independently", "checking math and capital cities"),
        ("Decide", "applying the calibrated threshold"),
    )
    tips = (
        "The gate asks for several answers so it can measure how much the model disagrees with itself.",
        "Agreement between samples is a signal, not proof that an answer is true.",
        "Only percentage math and capital cities are independently verified today.",
        "VERIFY does not mean wrong. It means the gate is not confident enough to recommend trust.",
    )

    pipe_html = ""
    if active is not None:
        rows = []
        for index, (label, hint) in enumerate(steps):
            if index < active or (index == 0 and model_ready):
                state, mark = "done", "&#10003;"
            elif index == active:
                state, mark = "active", ""
            else:
                state, mark = "todo", ""
            extra = ""
            if index == 1 and samples_total:
                pip_classes = []
                for n in range(samples_total):
                    if state == "done" or n < samples_done:
                        pip_classes.append("pip done")
                    elif n == samples_done and state == "active":
                        pip_classes.append("pip active")
                    else:
                        pip_classes.append("pip")
                extra = '<span class="pips">' + "".join(f'<i class="{cls}"></i>' for cls in pip_classes) + "</span>"
                if state != "todo":
                    hint = f"{min(samples_done, samples_total)} of {samples_total} answers written"
            rows.append(
                f'<li class="{state}"><span class="pd" aria-hidden="true">{mark}</span>'
                f"<span>{esc(label)}<small>{esc(hint)}</small></span>{extra}</li>"
            )
        pipe_html = f'<ol class="pipe">{"".join(rows)}</ol>'

    def safe_loading_text(value: object) -> str:
        return esc(str(value).replace("%", " percent"))

    loading_detail = str(detail)
    if "Still working" not in loading_detail:
        loading_detail = f"Still working · {loading_detail}"

    preview_html = ""
    shown = list(previews)[-5:]
    if shown:
        offset = len(previews) - len(shown)
        items = []
        for position, text in enumerate(shown, start=offset + 1):
            clean = " ".join(str(text).split())[:110] or "(empty answer)"
            clean = clean.replace("%", " percent")
            newest = " new" if position == len(previews) else ""
            items.append(f'<div class="pv{newest}"><b>{position:02d}</b><span>{esc(clean)}</span></div>')
        preview_html = '<div class="pv-label">Answers so far</div><div class="pvs">' + "".join(items) + "</div>"

    candidate_items = []
    candidate_shown = list(previews)[-5:]
    candidate_offset = len(previews) - len(candidate_shown)
    for position, text in enumerate(candidate_shown, start=candidate_offset + 1):
        clean = " ".join(str(text).split())[:80] or "(empty answer)"
        clean = clean.replace("%", " percent")
        newest = " new" if position == len(previews) else ""
        candidate_items.append(f'<span class="candidate{newest}">{esc(clean)}</span>')
    candidates_html = (
        '<div class="candidate-label">Candidate answers</div><div class="candidates">'
        + "".join(candidate_items) + "</div>"
    )
    clock_html = f'<span class="lclock" aria-hidden="true">{int(max(elapsed, 0.0))} s</span>'

    return f"""
    <div class="loading card" role="status" aria-live="polite">
        <div class="answer-skeleton" aria-hidden="true">
            <div>
                <div class="skeleton-heading"><span class="eyebrow">CONSENSUS ANSWER</span><span class="preparing-tag">PREPARING</span></div>
                <div class="skeleton-lines"><i class="skeleton-line"></i><i class="skeleton-line"></i><i class="skeleton-line"></i></div>
                <div class="thinking"><i></i><i></i><i></i><span>Thinking</span></div>
            </div>
            <div class="skeleton-gauge"><span>MEASURING</span></div>
        </div>
        <div class="lmain">
            <div class="lhead"><div class="ltitle">{safe_loading_text(title)}</div>{clock_html}</div>
            <div class="lsub">{safe_loading_text(loading_detail)}</div>
            {pipe_html}
            {candidates_html}
            {preview_html}
        </div>
        <div class="lbar" aria-hidden="true"></div>
    </div>
    """


def empty_state_html() -> str:
    return f"""
    <div class="steps">
        <div class="step card">
            <div class="step-num">1</div>
            <div class="step-title">Sample</div>
            <div class="step-text">Qwen3-0.6B writes {N_SAMPLES} independent answers to your question, locally.</div>
        </div>
        <div class="step card" style="animation-delay:0.1s">
            <div class="step-num">2</div>
            <div class="step-title">Measure</div>
            <div class="step-text">The gate checks how much the answers agree and how confident the model's tokens were.</div>
        </div>
        <div class="step card" style="animation-delay:0.2s">
            <div class="step-num">3</div>
            <div class="step-title">Decide</div>
            <div class="step-text">A calibrated model estimates the probability of correctness and recommends TRUST or VERIFY.</div>
        </div>
    </div>
    """


def gauge_svg(
    probability: float,
    threshold: float,
    decision: str,
    score_valid: bool = True,
    fresh: bool = False,
) -> str:
    """Draw the gauge. When score_valid is False no probability is drawn or printed."""
    cx = cy = 140
    radius = 100
    circumference = 2 * math.pi * radius
    plotted = probability if score_valid else 0.0
    value_text = f"{probability * 100:.1f}%" if score_valid else "N/A"
    caption_text = "EST. RELIABILITY" if score_valid else "NOT REPORTED"
    aria = (
        f"Estimated reliability {probability * 100:.1f} percent, trust boundary {pct_plain(threshold)} percent"
        if score_valid
        else "Reliability not reported for this response"
    )
    offset = circumference * (1 - plotted)

    zone_radius = radius + 9
    zone_circ = 2 * math.pi * zone_radius
    zone_dash = zone_circ * (1 - threshold)
    zone_start = zone_circ * threshold

    angle = 2 * math.pi * threshold

    def point(r: float) -> tuple[float, float]:
        return cx + r * math.sin(angle), cy - r * math.cos(angle)

    x1, y1 = point(radius - 15)
    x2, y2 = point(radius + 17)
    lx, ly = point(radius + 33)

    if score_valid:
        stop_a, stop_b = "var(--accent)", "var(--accent-2)"
    else:
        stop_a, stop_b = "var(--verify)", "var(--verify)"
    word_color = "var(--trust)" if decision == "TRUST" else "var(--verify)"
    ring_class = "ring-prog fresh-ring" if fresh and score_valid else "ring-prog"

    return f"""
    <svg class="gauge" viewBox="0 0 280 280" role="img"
         aria-label="{aria}">
        <defs>
            <linearGradient id="ringgrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="{stop_a}"/>
                <stop offset="1" stop-color="{stop_b}"/>
            </linearGradient>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="var(--surface)" stroke="var(--surface-2)" stroke-width="18"/>
        <circle cx="{cx}" cy="{cy}" r="{zone_radius}" fill="none" stroke="var(--trust)" stroke-opacity="0.4"
                stroke-width="5" stroke-linecap="round" stroke-dasharray="{zone_dash:.2f} {zone_circ:.2f}"
                stroke-dashoffset="{-zone_start:.2f}" transform="rotate(-90 {cx} {cy})"/>
        <circle class="{ring_class}" cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="url(#ringgrad)"
                stroke-width="18" stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}" style="--circ:{circumference:.2f};--off:{offset:.2f}"
                transform="rotate(-90 {cx} {cy})"/>
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="var(--text)" stroke-width="3" stroke-linecap="round"/>
        <text x="{lx:.2f}" y="{ly + 4:.2f}" text-anchor="middle" font-size="12" font-weight="700" fill="var(--muted)"
              style="font-family:inherit">{pct_plain(threshold)}%</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="50" font-weight="800" fill="var(--text)"
              style="font-family:inherit">{value_text}</text>
        <text x="{cx}" y="{cy + 40}" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="var(--muted)"
              style="font-family:inherit">{caption_text}</text>
        <text x="{cx}" y="{cy + 66}" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="2" fill="{word_color}"
              style="font-family:inherit">{decision}</text>
    </svg>
    """


def result_html(view: dict, fresh: bool = False) -> str:
    import urllib.parse

    decision = view["decision"]
    state = decision.lower()
    probability = view["probability"]
    threshold = view["threshold"]
    thr_text = pct_plain(threshold)
    comparator = "&ge;" if probability >= threshold else "&lt;"
    score_valid = view.get("score_valid", view["quality_ok"] and view.get("calibrated", True))

    if decision == "TRUST":
        icon = (
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            '<path d="M12 2.5l8 3v6.2c0 4.6-3.2 8.3-8 9.8-4.8-1.5-8-5.2-8-9.8V5.5l8-3z" stroke="var(--trust)" stroke-width="1.8"/>'
            '<path d="M8.2 12.2l2.6 2.6 5-5.4" stroke="var(--trust)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        message = (
            "<b>Trust recommended.</b> The sampled answers and token confidence "
            "cleared the gate's calibrated threshold. This is still an estimate, "
            "not proof."
        )
    else:
        icon = (
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            '<circle cx="12" cy="12" r="9" stroke="var(--verify)" stroke-width="2"/>'
            '<path d="M12 7.5v5.2M12 16.5h.01" stroke="var(--verify)" stroke-width="2.2" stroke-linecap="round"/></svg>'
        )
        if not view.get("calibrated", True):
            message = (
                "<b>Calibration refresh required.</b> This gate was not trained with the current "
                "prompt and generation settings, so its probability is not a calibrated estimate "
                "and every answer is shown as VERIFY."
                + ("" if view["quality_ok"] else f" Response-quality check: {esc(view['quality_reason'])}")
            )
        elif not view["quality_ok"]:
            message = (
                "<b>Response-quality safeguard triggered.</b> "
                f"{esc(view['quality_reason'])} The final recommendation is VERIFY. "
                "No reliability percentage is reported for a non-answer."
            )
        elif view.get("material_disagreement", False):
            message = (
                "<b>Sample disagreement detected.</b> The generated answers contain "
                "materially different useful claims, so the final recommendation is VERIFY. "
                "Agreement among samples is required before a reliability score can authorize trust."
            )
        elif view.get("fact_verification_status", "UNVERIFIED") == "UNSUPPORTED":
            message = ""
        elif view.get("fact_verification_status", "UNVERIFIED") != "VERIFIED":
            message = (
                "<b>Independent verification required.</b> "
                f"{esc(view.get('fact_verification_reason', 'The answer was not independently verified.'))} "
                "The final recommendation is VERIFY."
            )
        else:
            message = (
                "<b>Verification recommended.</b> The gate isn't confident enough to "
                "recommend trusting this answer. It may still be correct — check it "
                "against a reliable source."
            )

    if view["quality_ok"]:
        answer_text = esc(view["answer"]) if view["answer"] else "No answer text was produced."
        raw_output_html = ""
    else:
        answer_text = "No useful answer produced"
        raw_value = esc(view["answer"]) if view["answer"] else "(empty output)"
        raw_output_html = (
            '<div class="quality-raw"><span class="quality-label">Response quality check flagged</span>'
            '<div>The model output was not presented as a useful answer:</div>'
            f'<b>{raw_value}</b></div>'
        )

    reveal_words = 0
    if fresh:
        source_answer = str(view["answer"]) if view["quality_ok"] and view["answer"] else answer_text
        word_parts = re.split(r"(\s+)", source_answer)
        rendered_parts = []
        word_index = 0
        for part in word_parts:
            if part and not part.isspace():
                if word_index < 60:
                    rendered_parts.append(
                        f'<span class="answer-word" style="--i:{word_index}">{esc(part)}</span>'
                    )
                else:
                    rendered_parts.append(esc(part))
                word_index += 1
            else:
                rendered_parts.append(esc(part))
        answer_text = "".join(rendered_parts)
        reveal_words = min(word_index, 60)

    verification_status = view.get("fact_verification_status", "UNVERIFIED")
    verification_reason = view.get("fact_verification_reason", "No independent verification result.")
    verification_evidence = view.get("fact_verification_evidence", "")
    verification_url = view.get("fact_verification_source_url", "")
    unsupported = verification_status == "UNSUPPORTED"
    verification_class = "verified" if verification_status == "VERIFIED" else "failed"
    verification_heading = {
        "VERIFIED": "Independent verification: VERIFIED",
        "CONTRADICTED": "Independent verification: CONTRADICTED",
        "UNVERIFIED": "Independent verification: UNVERIFIED",
        "UNSUPPORTED": "Can't be independently verified",
        "UNAVAILABLE": "Independent verification: SOURCE UNAVAILABLE",
    }.get(verification_status, "Independent verification: UNVERIFIED")
    verification_link = (
        f'<div><a href="{esc(verification_url)}" target="_blank" rel="noopener noreferrer">Open reference source</a></div>'
        if verification_url and not unsupported else ""
    )
    if unsupported:
        verification_detail = (
            "Only percentage math and capital cities have a verifier, so the gate recommends "
            "VERIFY; this does not mean the answer is wrong."
        )
    else:
        verification_detail = esc(verification_reason)
        if verification_evidence:
            verification_detail += f" {esc(verification_evidence)}"
    verification_html = (
        f'<div class="verification-note {verification_class}"'
        f'{" data-status=\"UNSUPPORTED QUESTION TYPE\"" if unsupported else ""}>'
        f'<b>{esc(verification_heading)}</b><div>{verification_detail}</div>{verification_link}</div>'
    )

    calibrated = view.get("calibrated", True)
    if not calibrated:
        chip_text = "Calibration refresh required"
    elif unsupported:
        chip_text = ""
    elif verification_status != "VERIFIED":
        chip_text = "Independent verification required"
    elif not score_valid:
        chip_text = "Reliability not reported"
    else:
        chip_text = f"Trust threshold: {thr_text}%"

    if not view["quality_ok"]:
        comparison_caption = (
            'N/A — reliability not reported for a non-answer → <em class="verify">VERIFY</em>'
        )
    elif not calibrated:
        comparison_caption = (
            'N/A — calibration unavailable or unvalidated → <em class="verify">VERIFY</em>'
        )
    elif unsupported:
        comparison_caption = ""
    elif verification_status != "VERIFIED":
        comparison_caption = (
            'N/A — independent verification incomplete → <em class="verify">VERIFY</em>'
        )
    elif not score_valid:
        comparison_caption = 'N/A — reliability not reported → <em class="verify">VERIFY</em>'
    else:
        comparison_caption = (
            f"{probability * 100:.1f}% {comparator} {thr_text}% → "
            f'<em class="{state}">{decision}</em>'
        )

    if not view["quality_ok"]:
        disclaimer = "N/A — reliability not reported for a non-answer. The quality check only detects obvious response failures."
    elif not calibrated:
        disclaimer = "Reliability estimate unavailable: the loaded calibration is not validated for the current model, prompt and generation settings."
    elif unsupported:
        disclaimer = ""
    elif verification_status != "VERIFIED":
        disclaimer = "No reliability score is reported because this answer lacks an independent verification result. The verifier supports only a limited set of question types."
    else:
        disclaimer = (
            "Reliability is the gate's calibrated estimate of the probability that "
            "this answer is correct. It is a decision policy, not a guarantee of factual truth."
        )

    open_ended_html = ""
    if is_open_ended(view["question"]):
        open_ended_html = (
            '<div class="hint-note"><b>Open-ended request.</b>'
            "The gate is built for short factual questions (answers are capped at about 48 tokens), "
            "so long explanations or word-count requests cannot be scored reliably.</div>"
        )
    meta_html = f"Checked in {view.get('latency', 0.0):.1f} s · {esc(view.get('runtime', 'Unknown'))}"
    search_link = ""
    if decision == "VERIFY":
        query = urllib.parse.quote_plus(str(view["question"]))
        search_link = (
            f'<a class="verify-link" href="https://www.google.com/search?q={query}" '
            'target="_blank" rel="noopener noreferrer">Search the web for this</a>'
        )
    threshold_chip = f'<div class="thr-chip">{chip_text}</div>' if chip_text else ""
    decision_html = f'<div class="decision-msg">{message}</div>' if message else ""
    disclaimer_html = f'<div class="disclaimer">{disclaimer}</div>' if disclaimer else ""
    fresh_class = " fresh" if fresh else ""
    badge_delay = reveal_words * 45 if fresh else 0

    return f"""
    <div class="rc-wrap {state}{fresh_class}" style="--answer-delay:{badge_delay}ms">
        <div class="rc">
            <div>
                <div class="eyebrow">Consensus answer</div>
                <div class="answer">{answer_text}</div>
                {raw_output_html}
                {verification_html}
                {open_ended_html}
                <div class="badge-row">
                    <div class="badge {state}" role="status" aria-label="Final decision: {decision}">{icon}{decision}</div>
                    {threshold_chip}
                    {search_link}
                </div>
                {decision_html}
                <div class="meta-line">{meta_html}</div>
                <div class="qecho"><b>Question</b> · {esc(view["question"])}</div>
                {disclaimer_html}
            </div>
            <div class="gauge-col">
                <div class="eyebrow">Reliability estimate</div>
                {gauge_svg(probability, threshold, decision, score_valid, fresh)}
                <div class="meter">
                    <div class="meter-fill {state}" style="--w:{(probability * 100 if score_valid else 0.0):.2f}%"></div>
                    <div class="meter-mark" style="left:{threshold * 100:.2f}%"><span>{thr_text}%</span></div>
                </div>
                <div class="meter-cap">{comparison_caption}</div>
            </div>
        </div>
    </div>
    """


def metric_card(
    label: str,
    value: str,
    note: str,
    accent: str,
    bar: float | None = None,
    sub: str = "",
    delay: float = 0.0,
) -> str:
    bar_html = (
        f'<div class="mbar"><i style="--w:{min(max(bar, 0.0), 1.0) * 100:.1f}%"></i></div>'
        if bar is not None
        else ""
    )
    sub_html = f'<div class="msub">{esc(sub)}</div>' if sub else ""

    return f"""
    <div class="mcard" style="--metric-accent:{accent};animation-delay:{delay:.2f}s" title="{esc(note)}">
        <div class="mlabel">{esc(label)}</div>
        <div class="mvalue">{esc(value)}</div>
        {bar_html}
        {sub_html}
        <div class="mnote">{esc(note)}</div>
    </div>
    """


def signals_html(view: dict) -> str:
    n = len(view["samples"]) or N_SAMPLES
    max_entropy = math.log2(n) if n > 1 else 1.0

    cards = [
        metric_card(
            "Answer agreement",
            f"{view['agreement'] * 100:.0f}%",
            METRIC_NOTES["agreement"],
            "var(--accent)",
            bar=view["agreement"],
            sub=f"{round(view['agreement'] * n)} of {n} samples in the largest group",
            delay=0.0,
        ),
        metric_card(
            "Consensus margin",
            f"{view['margin'] * 100:.0f}%",
            METRIC_NOTES["margin"],
            "var(--accent)",
            bar=view["margin"],
            sub="Winner vs. runner-up",
            delay=0.06,
        ),
        metric_card(
            "Answer-key entropy",
            f"{view['entropy']:.3f}",
            METRIC_NOTES["entropy"],
            "var(--info)",
            bar=view["entropy"] / max_entropy,
            sub=f"Maximum {max_entropy:.2f} for {n} distinct answers",
            delay=0.12,
        ),
        metric_card(
            "Mean token log-probability",
            f"{view['mean_lp']:.3f}",
            METRIC_NOTES["mean_lp"],
            "var(--accent-2)",
            sub=f"≈ {token_probability(view['mean_lp']) * 100:.0f}% geometric-mean token probability",
            delay=0.18,
        ),
        metric_card(
            "Minimum token log-probability",
            f"{view['min_lp']:.3f}",
            METRIC_NOTES["min_lp"],
            "var(--accent-2)",
            sub=f"≈ {token_probability(view['min_lp']) * 100:.0f}% weakest-token probability",
            delay=0.24,
        ),
    ]

    return f'<div class="grid5">{"".join(cards)}</div>'


def samples_html(view: dict) -> str:
    samples = view["samples"]

    if not samples:
        return '<div class="s-banner">No sampled answers were returned.</div>'

    distinct = view["distinct"]
    n = len(samples)

    if distinct <= 1:
        banner = (
            f'<div class="s-banner">All {n} samples reduced to the same answer key.</div>'
        )
    else:
        banner = (
            f'<div class="s-banner split">The {n} samples produced <b>{distinct} distinct '
            "answer keys</b> — the model disagreed with itself.</div>"
        )

    cards = []
    for index, (answer, key) in enumerate(zip(samples, view["keys"]), start=1):
        is_consensus = key == view["winner_key"]
        css = "" if is_consensus else " diff"
        tag = (
            '<span class="tag same">Matches consensus</span>'
            if is_consensus
            else '<span class="tag diff">Different answer</span>'
        )
        shown_key = esc(key) if key else "(empty key)"

        cards.append(
            f"""
            <div class="scard{css}">
                <div class="snum">{index:02d}</div>
                <div class="sbody">
                    <div class="stext">{esc(answer) if answer else "(empty answer)"}</div>
                    <div class="skey">answer key · {shown_key}</div>
                </div>
                {tag}
            </div>
            """
        )

    footer = (
        '<div class="s-foot">Grouping uses the gate\'s own answer-key extractor, '
        "which approximates — but does not perfectly capture — semantic equivalence.</div>"
    )

    return f'{banner}<div class="sgrid">{"".join(cards)}</div>{footer}'


def technical_html(view: dict) -> str:
    items = [
        ("Local model", MODEL_NAME),
        ("Number of samples", str(N_SAMPLES)),
        ("Trust threshold", f"{view['threshold']:.2f} ({pct_plain(view['threshold'])}%)"),
        ("Reliability model", "ExtraTrees (calibrated for current settings)" if view.get("calibrated", True) else "ExtraTrees (NOT calibrated for current settings)"),
        ("Final decision", view["decision"]),
        ("Classifier decision", view["model_decision"]),
        ("Response quality", "Passed" if view["quality_ok"] else "Flagged: " + view["quality_reason"]),
        ("Material sample disagreement", "Yes" if view.get("material_disagreement") else "No"),
        ("Independent fact verification", view.get("fact_verification_status", "UNVERIFIED")),
        ("Verifier result", view.get("fact_verification_reason", "No result")),
        ("Verifier evidence", view.get("fact_verification_evidence", "") or "No evidence"),
        ("Verifier source", view.get("fact_verification_source_url", "") or "No external source"),
        ("Runtime", view["runtime"]),
        ("Reliability reported", "Yes" if view.get("score_valid") else "No (N/A)"),
        ("Raw classifier score (diagnostic only, not a reliability estimate)" if not view.get("score_valid") else "Raw P(correct)", f"{view['probability']:.4f}"),
        ("Latency (this run)", f"{view['latency']:.1f} s"),
        ("Generated at", view["timestamp"]),
    ]

    kv = "".join(
        f"<div class='kv'><small>{esc(label)}</small><b>{esc(value)}</b></div>"
        for label, value in items
    )
    chips = "".join(f"<span class='chip'>{esc(name)}</span>" for name in EXPECTED_FEATURES)

    return f"""
    <div class="kvgrid">{kv}</div>
    <div class="feat-title">Reliability features used by the gate ({len(EXPECTED_FEATURES)})</div>
    <div class="chips">{chips}</div>
    """


def sidebar_html(threshold: float | None, history: list[dict]) -> str:
    if threshold is not None:
        thr_text = pct_plain(threshold)
        thr_display = f"{thr_text}%"
        policy_html = (
            '<div class="policy"><small>Decision policy</small>'
            f'P(correct) &ge; {thr_text}% → <span class="t">TRUST</span><br>'
            f'P(correct) &lt; {thr_text}% → <span class="v">VERIFY</span></div>'
        )
    else:
        thr_text = ""
        thr_display = "Not available yet"
        policy_html = (
            '<div class="policy"><small>Decision policy</small>'
            "Shown once a calibrated gate is loaded. Until then every answer is VERIFY.</div>"
        )

    history_html = ""
    if history:
        rows = "".join(
            f"""
            <div class="hist-item">
                <i class="{item['decision'].lower()}"></i>
                <em title="{esc(item['question'])}">{esc(item['question'])}</em>
                <b>{'N/A' if not item.get('score_valid', False) else f"{item['probability'] * 100:.0f}%"}</b>
            </div>
            """
            for item in history
        )
        history_html = f"""
        <div class="hist">
            <div class="eyebrow">This session</div>
            {rows}
        </div>
        """

    return f"""
    <div class="eyebrow">About the system</div>
    <div class="side-text">
        This system runs a small local LLM, samples multiple answers, extracts uncertainty
        signals, and uses a calibrated reliability model to decide whether the answer should
        be trusted or verified.
    </div>
    <div class="side-kv"><span>Model</span><b>{esc(MODEL_NAME)}</b></div>
    <div class="side-kv"><span>Samples</span><b>{N_SAMPLES}</b></div>
    <div class="side-kv"><span>Threshold</span><b>{thr_display}</b></div>
    {policy_html}
    <div class="side-note">
        <b>VERIFY does not mean the answer is definitely wrong.</b>
        It means the gate is not confident enough to recommend trusting it.
    </div>
    {history_html}
    """


# ============================================================
# SESSION STATE + CALLBACKS
# ============================================================

st.session_state.setdefault("question_input", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("fresh_result", False)
st.session_state.setdefault("history", [])
st.session_state.setdefault("auto_run", False)
if "theme_choice" not in st.session_state:
    query_theme = str(st.query_params.get("theme", "daylight")).strip().lower()
    initial_theme = query_theme if query_theme in THEMES else "daylight"
    st.session_state["theme_choice"] = THEMES[initial_theme]["label"]
active_theme = str(st.session_state["theme_choice"]).strip().lower()
if active_theme not in THEMES:
    active_theme = "daylight"
    st.session_state["theme_choice"] = THEMES[active_theme]["label"]


def update_theme_query() -> None:
    st.query_params["theme"] = str(st.session_state["theme_choice"]).strip().lower()


def use_example(example: str) -> None:
    """An example chip fills the box and runs the check in one click."""
    st.session_state["question_input"] = example
    st.session_state["auto_run"] = True


def clear_all() -> None:
    st.session_state["question_input"] = ""
    st.session_state["result"] = None
    st.session_state["fresh_result"] = False
    st.session_state["auto_run"] = False


# ============================================================
# PAGE
# ============================================================

st.html(CSS)
st.html(theme_css(active_theme))

theme_spacer, theme_column = st.columns([5, 2])
with theme_column:
    try:
        st.segmented_control(
            "Theme",
            [THEMES[name]["label"] for name in THEMES],
            key="theme_choice",
            label_visibility="collapsed",
            on_change=update_theme_query,
        )
    except AttributeError:
        st.radio(
            "Theme",
            [THEMES[name]["label"] for name in THEMES],
            key="theme_choice",
            horizontal=True,
            label_visibility="collapsed",
            on_change=update_theme_query,
        )

# Sidebar is painted immediately (not at the end of the script) so the page never looks half loaded.
sidebar_slot = st.sidebar.empty()


def paint_sidebar() -> None:
    state = process_state()
    sidebar_slot.markdown(
        html_block(
            sidebar_html(
                state["threshold"] if state["calibrated"] else None,
                st.session_state["history"],
            )
        ),
        unsafe_allow_html=True,
    )


paint_sidebar()

# The page renders immediately. Qwen is NOT loaded here; see the run handler below.
_state = process_state()
load_error = _state["error"]

hero_slot = st.empty()
_hero_status = "online" if _state["loaded"] else ("offline" if _state["error"] else "idle")
hero_slot.markdown(
    html_block(hero_html(_hero_status, _state["runtime"] if _state["loaded"] else None)),
    unsafe_allow_html=True,
)

calibration_slot = st.empty()
calibration_banner = calibration_banner_html(
    _state["loaded"], _state["calibrated"], _state["calibration_note"]
)
if calibration_banner:
    calibration_slot.markdown(html_block(calibration_banner), unsafe_allow_html=True)


# ---------------- compact input card ----------------

try:
    input_box = st.container(key="input_card")
except TypeError:  # older Streamlit without container keys
    input_box = st.container(border=True)

with input_box:
    render(
        """
        <div class="ask-head">
            <b>Ask the model</b>
            <span>Short factual questions work best. The gate samples several answers and estimates whether you can rely on them.</span>
        </div>
        """
    )

    with st.form("reliability_form", clear_on_submit=False):
        try:
            input_column, button_column = st.columns([5, 1.3], vertical_alignment="bottom")
        except TypeError:  # older Streamlit without vertical_alignment
            input_column, button_column = st.columns([5, 1.3])
        with input_column:
            st.text_input(
                "Question",
                key="question_input",
                placeholder="Ask a short factual question...",
                label_visibility="collapsed",
            )
        with button_column:
            run_clicked = st.form_submit_button(
                "Check Reliability",
                type="primary",
                use_container_width=True,
            )

    render('<div class="mini-label">Try an example</div>')

    try:
        example_chip_box = st.container(key="example_chips")
    except TypeError:  # older Streamlit without keyed containers
        example_chip_box = st.container()
    with example_chip_box:
        chip_columns = st.columns(len(EXAMPLE_QUESTIONS) + 1)
        for column, example in zip(chip_columns, EXAMPLE_QUESTIONS):
            with column:
                st.button(
                    example,
                    key=f"example_{example}",
                    on_click=use_example,
                    args=(example,),
                    use_container_width=True,
                )
        with chip_columns[-1]:
            st.button("Clear", key="clear_button", on_click=clear_all, use_container_width=True)

# An example chip sets auto_run in its callback; consume it exactly once.
run_clicked = bool(run_clicked) or bool(st.session_state.pop("auto_run", False))


# ---------------- run the gate (only on the submit rerun) ----------------

notice_slot = st.empty()
status_slot = st.empty()

if run_clicked:
    question_text = st.session_state.get("question_input", "").strip()
    state = process_state()

    if not question_text:
        notice_slot.markdown(
            html_block(
                notice_html(
                    "warn",
                    "Enter a question first",
                    "Type a factual question above, or pick one of the examples.",
                )
            ),
            unsafe_allow_html=True,
        )

    else:
        st.session_state["result"] = None
        st.session_state["fresh_result"] = False
        calibration_slot.empty()

        def show_wait(waited: float) -> None:
            status_slot.markdown(
                html_block(
                    loading_html(
                        "Waiting for another check to finish",
                        "Still working · only one generation runs on the GPU at a time.",
                        elapsed=waited,
                    )
                ),
                unsafe_allow_html=True,
            )

        if not acquire_run_lock(state, show_wait):
            status_slot.empty()
            notice_slot.markdown(
                html_block(
                    notice_html(
                        "warn",
                        "A check is already running",
                        "Wait for the current check to finish, then try again.",
                    )
                ),
                unsafe_allow_html=True,
            )
        else:
            try:
                gate = None
                new_view = None
                hero_slot.markdown(html_block(hero_html("loading")), unsafe_allow_html=True)

                live = {
                    "stage": "",
                    "done": 0,
                    "previews": [],
                    "t0": perf_counter(),
                    "ready": bool(state["loaded"]),
                }

                def paint() -> None:
                    stage = live["stage"]
                    active = stage_step(stage)
                    if live["done"] and active == 1:
                        detail = f"Still working · {live['done']} of {N_SAMPLES} answers written."
                    else:
                        detail = "Still working · local inference can take a while."
                    status_slot.markdown(
                        html_block(
                            loading_html(
                                stage,
                                detail,
                                active=active,
                                samples_done=live["done"],
                                samples_total=N_SAMPLES,
                                previews=live["previews"],
                                elapsed=perf_counter() - live["t0"],
                                model_ready=live["ready"],
                            )
                        ),
                        unsafe_allow_html=True,
                    )

                def show_stage(stage: str) -> None:
                    live["stage"] = stage
                    if stage.startswith("Sampling"):
                        live["done"] = 0
                        live["previews"] = []
                    paint()

                def show_progress(done: int, total: int, text: str) -> None:
                    live["done"] = done
                    live["previews"].append(text)
                    paint()

                try:
                    gate, load_error, new_view = load_and_run(
                        state["loaded"], question_text, show_stage, show_progress
                    )
                    if gate is not None and new_view is None:
                        raise ValueError("The reliability gate returned no result.")
                except (AttributeError, TypeError, ValueError, KeyError) as exc:
                    logger.exception("Gate returned an unexpected result")
                    gate = None
                    st.session_state["result"] = None
                    load_error = {
                        "title": "Unexpected result",
                        "body": "The gate returned a result the interface could not read. Please try again.",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                except Exception as exc:
                    logger.exception("Inference failed")
                    gate = None
                    st.session_state["result"] = None

                    if "out of memory" in str(exc).lower():
                        title = "GPU memory is full"
                        body = (
                            "The 4 GB GPU ran out of memory during generation. Close other "
                            "GPU-heavy applications and try again."
                        )
                        if torch is not None:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    else:
                        title = "The gate could not finish"
                        body = (
                            "Something went wrong while generating or scoring the answers. "
                            "Please try again."
                        )
                    load_error = {
                        "title": title,
                        "body": body,
                        "detail": f"{type(exc).__name__}: {exc}",
                    }

                if state["loaded"] and state["calibrated"]:
                    calibration_slot.empty()
                elif state["loaded"]:
                    calibration_slot.markdown(
                        html_block(
                            notice_html(
                                "warn",
                                "Calibration refresh required",
                                state["calibration_note"],
                            )
                        ),
                        unsafe_allow_html=True,
                    )

                if gate is None:
                    hero_status = "online" if state["loaded"] else "offline"
                    hero_slot.markdown(
                        html_block(hero_html(hero_status, state["runtime"])),
                        unsafe_allow_html=True,
                    )
                    notice_slot.markdown(
                        html_block(
                            notice_html(
                                "error",
                                (load_error or {}).get("title", "Model unavailable"),
                                (load_error or {}).get("body", "The local model is not available."),
                            )
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    hero_slot.markdown(
                        html_block(hero_html("online", runtime_label(gate))),
                        unsafe_allow_html=True,
                    )
                    st.session_state["result"] = new_view
                    st.session_state["history"] = (
                        [
                            {
                                "question": new_view["question"],
                                "decision": new_view["decision"],
                                "probability": new_view["probability"],
                                "score_valid": new_view["score_valid"],
                            }
                        ]
                        + st.session_state["history"]
                    )[:HISTORY_LIMIT]
                    st.session_state["fresh_result"] = True

            finally:
                # Release FIRST. When a rerun (Clear, new question) interrupts this script,
                # Streamlit raises from the next st.* call, including the one below. If that
                # call came first the lock would never be released and every later check
                # would report "A check is already running".
                state["lock"].release()
                status_slot.empty()

            paint_sidebar()


# ---------------- model-load error detail ----------------

if load_error is not None and not run_clicked:
    render(notice_html("error", load_error["title"], load_error["body"]))
    with st.expander("Error message"):
        st.code(load_error["detail"], language="text")


# ---------------- results / empty state ----------------

current = st.session_state.get("result")

if current is None:
    if load_error is None:
        render(empty_state_html())
else:
    fresh_result = bool(st.session_state.get("fresh_result", False))
    render(result_html(current, fresh=fresh_result))
    if fresh_result:
        st.session_state["fresh_result"] = False

    render(
        """
        <div class="section-title">Why did the gate decide this?</div>
        <div class="section-sub">
            Model signals extracted from the sampled answers. Hover a card for its definition.
        </div>
        """
    )
    render(signals_html(current))

    render('<div style="height:26px"></div>')

    with st.expander(f"See the {len(current['samples']) or N_SAMPLES} sampled answers"):
        render(samples_html(current))

    with st.expander("Technical details"):
        render(technical_html(current))

render(
    '<div class="foot">Everything runs locally · Reliability is an estimate, '
    "not a guarantee of truth</div>"
)
