from __future__ import annotations

import html
import logging
import math
import sys
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


from src.gate.agreement import extract_answer_key
from src.gate.response_quality import assess_response_quality
from src.gate.reliability_gate import (
    EXPECTED_FEATURES,
    MODEL_NAME,
    N_SAMPLES,
    ReliabilityGate,
)

try:
    import torch
except Exception:  # torch is only used to report runtime status
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

CSS = """
<style>
:root {
    --ink: #0f172a;
    --muted: #64748b;
    --line: rgba(99, 102, 241, 0.16);
    --indigo: #4f46e5;
    --violet: #7c3aed;
    --cyan: #06b6d4;
    --trust: #059669;
    --verify: #b45309;
    --shadow-1: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(79, 70, 229, 0.08);
    --shadow-2: 0 2px 4px rgba(15, 23, 42, 0.04), 0 14px 34px rgba(79, 70, 229, 0.12), 0 34px 70px rgba(79, 70, 229, 0.10);
}

html, body, .stApp {
    color-scheme: light;
    font-family: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
}

.stApp {
    color: var(--ink);
    background:
        radial-gradient(circle at 1px 1px, rgba(99, 102, 241, 0.09) 1px, transparent 0) 0 0 / 30px 30px,
        linear-gradient(160deg, #f8faff 0%, #f1f4ff 45%, #eefbff 100%);
}

[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
[data-testid="InputInstructions"] { display: none; }

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

/* ---------- background orbs ---------- */
.orbs { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.orb { position: absolute; border-radius: 50%; filter: blur(70px); opacity: 0.55; animation: drift 20s ease-in-out infinite; }
.orb.a { width: 420px; height: 420px; background: #c7d2fe; top: -140px; left: -90px; }
.orb.b { width: 380px; height: 380px; background: #a5f3fc; top: 32%; right: -130px; animation-duration: 26s; }
.orb.c { width: 340px; height: 340px; background: #ddd6fe; bottom: -140px; left: 38%; animation-duration: 23s; }

@keyframes drift {
    0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
    50% { transform: translate3d(30px, -26px, 0) scale(1.08); }
}
@keyframes bob {
    0%, 100% { transform: translateY(0) rotate(-1deg); }
    50% { transform: translateY(-6px) rotate(1deg); }
}
@keyframes ping {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.45); }
    80%, 100% { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
}
@keyframes rise {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes flow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}
@keyframes ringfill {
    from { stroke-dashoffset: var(--circ); }
    to { stroke-dashoffset: var(--off); }
}
@keyframes grow {
    from { width: 0; }
    to { width: var(--w); }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes wave {
    0%, 100% { opacity: 0.35; transform: translateX(0); }
    50% { opacity: 1; transform: translateX(4px); }
}
@keyframes slide {
    0% { left: -40%; }
    100% { left: 100%; }
}

/* ---------- hero ---------- */
.hero { display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap; padding: 4px 2px 20px; }
.brand { display: flex; align-items: center; gap: 16px; }
.logo {
    width: 62px; height: 62px; border-radius: 19px; display: grid; place-items: center;
    background: linear-gradient(145deg, #ffffff, #e9edff);
    border: 1px solid rgba(255, 255, 255, 0.95);
    box-shadow: inset 0 1px 0 #fff, 0 12px 26px rgba(79, 70, 229, 0.28), 0 3px 8px rgba(79, 70, 229, 0.14);
    animation: bob 6s ease-in-out infinite;
}
.eyebrow { font-size: 0.7rem; letter-spacing: 0.16em; font-weight: 700; color: var(--indigo); text-transform: uppercase; }
.title {
    font-size: 2.1rem; line-height: 1.12; font-weight: 800; letter-spacing: -0.02em; margin: 3px 0 2px;
    background: linear-gradient(90deg, #0f172a, #312e81 55%, #4f46e5);
    -webkit-background-clip: text; background-clip: text;
    color: transparent; -webkit-text-fill-color: transparent;
}
.subtitle { color: var(--muted); font-size: 1.02rem; }
.status-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px;
    background: rgba(255, 255, 255, 0.82); border: 1px solid var(--line); box-shadow: var(--shadow-1);
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; color: #334155; backdrop-filter: blur(8px);
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: ping 2s infinite; }
.dot.loading { background: #f59e0b; animation: none; }
.dot.off { background: #ef4444; animation: none; }

/* ---------- input card ---------- */
.st-key-input_card {
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.95);
    outline: 1px solid var(--line);
    border-radius: 26px;
    padding: 26px 28px 22px;
    box-shadow: var(--shadow-2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.st-key-input_card:hover { transform: translateY(-3px); }
.card-head { font-size: 1.35rem; font-weight: 750; letter-spacing: -0.01em; margin-bottom: 2px; }
.card-sub { color: var(--muted); font-size: 0.92rem; margin-bottom: 4px; }
.mini-label { font-size: 0.68rem; letter-spacing: 0.14em; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin: 2px 0 -6px; }

.stTextArea textarea { font-size: 1.05rem; line-height: 1.5; color: var(--ink); background: #ffffff; }
.stTextArea [data-baseweb="textarea"] {
    border-radius: 16px; border: 1.5px solid rgba(99, 102, 241, 0.22); background: #ffffff;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextArea [data-baseweb="base-input"] { background: #ffffff; border-radius: 16px; }
.stTextArea [data-baseweb="textarea"]:focus-within {
    border-color: #6366f1;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.14), 0 10px 26px rgba(79, 70, 229, 0.12);
}

/* ---------- buttons: secondary = chips, primary = CTA ---------- */
.stButton > button {
    border-radius: 999px; border: 1px solid rgba(99, 102, 241, 0.22); background: rgba(255, 255, 255, 0.92);
    color: #334155; font-weight: 600; min-height: 2.6rem; padding: 0.5rem 1rem;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 5px 14px rgba(79, 70, 229, 0.07);
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease, color 0.22s ease;
}
.stButton > button:hover {
    transform: translateY(-3px); border-color: #6366f1; color: #4338ca;
    box-shadow: 0 10px 24px rgba(79, 70, 229, 0.18);
}
.stButton > button p { font-size: 0.84rem; line-height: 1.25; margin: 0; white-space: normal; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    min-height: 3.3rem; border: none; color: #ffffff;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 120%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 12px 28px rgba(79, 70, 229, 0.38);
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    color: #ffffff; transform: translateY(-3px) scale(1.01);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.4), 0 18px 38px rgba(79, 70, 229, 0.45);
}
.stButton > button[kind="primary"] p,
.stButton > button[data-testid="stBaseButton-primary"] p { color: #ffffff; font-size: 1.02rem; font-weight: 700; letter-spacing: 0.01em; }

/* ---------- generic cards ---------- */
.card {
    background: rgba(255, 255, 255, 0.86); backdrop-filter: blur(14px); border: 1px solid var(--line);
    border-radius: 22px; box-shadow: var(--shadow-1);
}
.section-title { font-size: 1.28rem; font-weight: 750; letter-spacing: -0.01em; margin: 38px 0 2px; }
.section-sub { color: var(--muted); font-size: 0.92rem; margin-bottom: 16px; }

/* ---------- notices ---------- */
.notice { display: flex; gap: 12px; align-items: flex-start; padding: 14px 18px; border-radius: 16px; margin-top: 16px; border: 1px solid; animation: rise 0.4s ease both; }
.notice.warn { background: #fffbeb; border-color: #fde68a; color: #78350f; }
.notice.error { background: #fff1f2; border-color: #fecdd3; color: #881337; }
.notice b { display: block; margin-bottom: 2px; }
.notice span { font-size: 0.92rem; }

/* ---------- loading ---------- */
.loading { position: relative; overflow: hidden; display: flex; gap: 22px; align-items: center; padding: 26px 28px; margin-top: 24px; box-shadow: var(--shadow-2); animation: rise 0.4s ease both; }
.spin { flex: none; width: 54px; height: 54px; border-radius: 50%; border: 5px solid #e0e7ff; border-top-color: #6366f1; border-right-color: #06b6d4; animation: spin 1s linear infinite; }
.ltitle { font-size: 1.15rem; font-weight: 750; }
.lsub { color: var(--muted); font-size: 0.9rem; margin-bottom: 10px; }
.lsteps { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.lsteps li { font-size: 0.93rem; color: #334155; display: flex; gap: 10px; align-items: center; animation: wave 2.4s ease-in-out infinite; animation-delay: var(--d); }
.lsteps li::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: linear-gradient(135deg, #6366f1, #06b6d4); }
.lbar { position: absolute; left: 0; right: 0; bottom: 0; height: 4px; background: #eef2ff; }
.lbar::after { content: ""; position: absolute; top: 0; bottom: 0; width: 40%; border-radius: 4px; background: linear-gradient(90deg, #6366f1, #06b6d4); animation: slide 1.6s ease-in-out infinite; }

/* ---------- empty state ---------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-top: 26px; }
.step { position: relative; padding: 22px; transition: transform 0.3s ease, box-shadow 0.3s ease; animation: rise 0.6s ease both; }
.step:hover { transform: translateY(-6px); box-shadow: var(--shadow-2); }
.step-num { width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; font-weight: 800; color: #fff; margin-bottom: 12px; background: linear-gradient(135deg, #4f46e5, #06b6d4); box-shadow: 0 8px 18px rgba(79, 70, 229, 0.3); }
.step-title { font-weight: 750; margin-bottom: 4px; }
.step-text { color: var(--muted); font-size: 0.9rem; line-height: 1.5; }

/* ---------- result card ---------- */
.rc-wrap {
    --c1: #10b981; --c2: #06b6d4;
    margin: 26px 0 40px; padding: 2px; border-radius: 30px;
    background: linear-gradient(120deg, var(--c1), var(--c2), var(--c1)); background-size: 200% 200%;
    animation: flow 7s linear infinite, rise 0.6s ease both;
    box-shadow: 0 12px 0 -6px rgba(255, 255, 255, 0.8), 0 24px 0 -12px rgba(255, 255, 255, 0.55), 0 34px 70px rgba(79, 70, 229, 0.16);
    transition: transform 0.35s ease;
}
.rc-wrap:hover { transform: translateY(-5px); }
.rc-wrap.verify { --c1: #f59e0b; --c2: #fb923c; }
.rc {
    display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 30px; padding: 32px;
    border-radius: 28px; background: linear-gradient(180deg, #ffffff, #fafbff);
}
@media (max-width: 860px) { .rc { grid-template-columns: minmax(0, 1fr); padding: 24px; } }
.qecho { color: var(--muted); font-size: 0.92rem; margin-bottom: 18px; word-break: break-word; }
.qecho b { color: #334155; }
.answer { font-size: 1.9rem; line-height: 1.25; font-weight: 750; letter-spacing: -0.015em; margin: 6px 0 22px; word-break: break-word; }
.badge-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.badge { display: inline-flex; align-items: center; gap: 10px; padding: 12px 22px; border-radius: 16px; font-weight: 800; font-size: 1.25rem; letter-spacing: 0.06em; }
.badge.trust { color: #fff; background: linear-gradient(135deg, #10b981, #059669); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35), 0 12px 26px rgba(5, 150, 105, 0.35); }
.badge.verify { color: #451a03; background: linear-gradient(135deg, #fcd34d, #f59e0b); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5), 0 12px 26px rgba(245, 158, 11, 0.38); }
.thr-chip { padding: 8px 14px; border-radius: 12px; background: #f1f5ff; border: 1px solid var(--line); font-size: 0.85rem; font-weight: 650; color: #334155; }
.decision-msg { color: #334155; font-size: 0.97rem; line-height: 1.55; margin-bottom: 14px; }
.disclaimer { color: var(--muted); font-size: 0.8rem; line-height: 1.5; border-top: 1px dashed rgba(99, 102, 241, 0.25); padding-top: 12px; }

.gauge-col { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.gauge { width: 100%; max-width: 290px; filter: drop-shadow(0 16px 24px rgba(79, 70, 229, 0.16)); }
.ring-prog { animation: ringfill 1.5s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.meter { position: relative; width: 100%; max-width: 320px; height: 12px; border-radius: 99px; background: #e8ecfb; margin: 22px 0 8px; }
.meter-fill { height: 100%; border-radius: 99px; width: var(--w); animation: grow 1.4s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.meter-fill.trust { background: linear-gradient(90deg, #34d399, #06b6d4); }
.meter-fill.verify { background: linear-gradient(90deg, #fcd34d, #f97316); }
.meter-mark { position: absolute; top: -6px; width: 3px; height: 24px; border-radius: 3px; background: #0f172a; transform: translateX(-1px); }
.meter-mark span { position: absolute; top: 28px; left: 50%; transform: translateX(-50%); font-size: 0.7rem; font-weight: 700; color: #475569; }
.meter-cap { margin-top: 22px; font-size: 0.95rem; font-weight: 700; color: #334155; text-align: center; }
.meter-cap em { font-style: normal; }
.meter-cap .trust { color: var(--trust); }
.meter-cap .verify { color: var(--verify); }

/* ---------- metric cards ---------- */
.grid5 { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; }
.mcard {
    --accent: #6366f1; position: relative; overflow: hidden; padding: 18px 18px 16px; background: #fff;
    border: 1px solid var(--line); border-radius: 20px; box-shadow: var(--shadow-1);
    transition: transform 0.28s ease, box-shadow 0.28s ease; animation: rise 0.6s ease both; cursor: default;
}
.mcard::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: linear-gradient(90deg, var(--accent), transparent); }
.mcard:hover { transform: translateY(-7px); box-shadow: var(--shadow-2); }
.mlabel { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.06em; color: #64748b; text-transform: uppercase; min-height: 2.1em; }
.mvalue { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin: 6px 0 8px; font-variant-numeric: tabular-nums; }
.mbar { height: 7px; border-radius: 99px; background: #eef2ff; margin-bottom: 10px; overflow: hidden; }
.mbar i { display: block; height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg, var(--accent), #06b6d4); animation: grow 1.2s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.msub { font-size: 0.78rem; font-weight: 650; color: #334155; margin-bottom: 6px; }
.mnote { font-size: 0.8rem; color: var(--muted); line-height: 1.45; }

/* ---------- samples ---------- */
.s-banner { padding: 12px 16px; border-radius: 14px; background: #eef2ff; border: 1px solid var(--line); font-size: 0.9rem; color: #334155; margin-bottom: 14px; }
.s-banner.split { background: #fffbeb; border-color: #fde68a; color: #78350f; }
.sgrid { display: grid; gap: 12px; }
.scard {
    display: flex; gap: 16px; align-items: flex-start; padding: 14px 18px; background: #fff;
    border: 1px solid var(--line); border-left: 4px solid #6366f1; border-radius: 16px;
    box-shadow: var(--shadow-1); transition: transform 0.22s ease, box-shadow 0.22s ease;
}
.scard.diff { border-left-color: #f59e0b; background: #fffdf7; }
.scard:hover { transform: translateX(5px); box-shadow: var(--shadow-2); }
.snum { font-size: 1.4rem; font-weight: 800; color: #a5b4fc; font-variant-numeric: tabular-nums; line-height: 1.2; }
.scard.diff .snum { color: #fbbf24; }
.sbody { min-width: 0; flex: 1; }
.stext { font-size: 1rem; font-weight: 600; word-break: break-word; }
.skey { margin-top: 4px; font-size: 0.76rem; color: #64748b; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; word-break: break-word; }
.tag { flex: none; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; padding: 5px 10px; border-radius: 99px; }
.tag.same { background: #e0e7ff; color: #3730a3; }
.tag.diff { background: #fef3c7; color: #92400e; }
.s-foot { margin-top: 12px; font-size: 0.78rem; color: var(--muted); }

/* ---------- technical details ---------- */
.kvgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
.kv { padding: 12px 16px; border-radius: 14px; background: #f8faff; border: 1px solid var(--line); }
.kv small { display: block; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: #94a3b8; font-weight: 700; margin-bottom: 3px; }
.kv b { font-size: 0.95rem; word-break: break-word; }
.feat-title { margin: 20px 0 10px; font-size: 0.8rem; font-weight: 700; color: #475569; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { padding: 5px 11px; border-radius: 99px; background: #eef2ff; color: #3730a3; font-size: 0.74rem; font-weight: 600; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; }

/* ---------- expanders ---------- */
[data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: 18px; background: rgba(255, 255, 255, 0.88);
    box-shadow: var(--shadow-1); margin-bottom: 14px; overflow: hidden; transition: box-shadow 0.25s ease;
}
[data-testid="stExpander"]:hover { box-shadow: var(--shadow-2); }
[data-testid="stExpander"] summary { padding: 0.9rem 1.1rem; font-weight: 700; color: var(--ink); }
[data-testid="stExpander"] summary p { font-size: 1rem; font-weight: 700; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #f3f6ff 100%); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
.side-text { font-size: 0.9rem; line-height: 1.55; color: #475569; margin: 8px 0 16px; }
.side-kv { display: flex; justify-content: space-between; gap: 10px; padding: 10px 0; border-bottom: 1px solid rgba(99, 102, 241, 0.12); font-size: 0.86rem; }
.side-kv span { color: #64748b; }
.side-kv b { text-align: right; }
.policy { margin: 16px 0; padding: 14px; border-radius: 16px; background: #fff; border: 1px solid var(--line); box-shadow: var(--shadow-1); font-size: 0.85rem; line-height: 1.9; }
.policy .t { color: var(--trust); font-weight: 800; }
.policy .v { color: var(--verify); font-weight: 800; }
.policy small { display: block; font-size: 0.66rem; letter-spacing: 0.14em; color: #94a3b8; font-weight: 700; text-transform: uppercase; line-height: 1.4; margin-bottom: 4px; }
.side-note { padding: 13px 14px; border-radius: 14px; background: #fffbeb; border: 1px solid #fde68a; color: #78350f !important; font-size: 0.8rem; line-height: 1.5; }
.hist { margin-top: 22px; }
.hist-item { display: flex; gap: 10px; align-items: center; padding: 8px 0; font-size: 0.82rem; border-bottom: 1px solid rgba(99, 102, 241, 0.1); }
.hist-item i { flex: none; width: 8px; height: 8px; border-radius: 50%; }
.hist-item i.trust { background: #10b981; }
.hist-item i.verify { background: #f59e0b; }
.hist-item em { flex: 1; font-style: normal; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hist-item b { flex: none; font-variant-numeric: tabular-nums; }

.foot { text-align: center; color: #94a3b8; font-size: 0.78rem; margin-top: 10px; }

@media (prefers-reduced-motion: reduce) {
    * { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
}
.quality-raw { margin: 2px 0 16px; padding: 12px 14px; border-radius: 12px; background: #fff8e7; border: 1px solid #f4d58d; color: #78350f; font-size: 0.88rem; line-height: 1.5; }
.quality-raw b { display: block; margin-top: 3px; color: #451a03; word-break: break-word; }
.quality-label { display: inline-flex; margin-bottom: 8px; padding: 5px 9px; border-radius: 99px; background: #fff1c2; color: #92400e; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
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


# ============================================================
# MODEL LOADING (once per process)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_gate() -> ReliabilityGate:
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
                "The trained gate file could not be found. Make sure "
                "artifacts/classifier/reliability_gate_final.pkl exists "
                "(it is created by scripts/finalize_gate_v3.py)."
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

def answer_groups(question: str, answers: list[str]) -> tuple[list[str], str, int]:
    """Group samples with the gate's own answer-key extractor."""
    keys: list[str] = []

    for answer in answers:
        try:
            key = extract_answer_key(question, answer)
        except Exception:
            key = " ".join(answer.lower().split())
        keys.append(key)

    if not keys:
        return keys, "", 0

    counts = Counter(keys)
    return keys, counts.most_common(1)[0][0], len(counts)


def build_view(result: object, latency: float, runtime: str) -> dict:
    """Validate the GateResult and convert it into a plain dict for rendering."""
    probability = float(result.probability_correct)
    threshold = float(result.threshold)

    if not (math.isfinite(probability) and math.isfinite(threshold)):
        raise ValueError("Non-finite probability or threshold.")

    probability = min(max(probability, 0.0), 1.0)

    model_decision = str(result.decision).strip().upper()
    if model_decision not in {"TRUST", "VERIFY"}:
        model_decision = "TRUST" if probability >= threshold else "VERIFY"

    samples = [str(answer) for answer in (result.raw_answers or [])]
    question = str(result.question)
    answer = str(result.answer or "").strip()
    quality = assess_response_quality(question, answer)
    # A basic-usefulness failure can only make the display decision more cautious.
    # The ML probability is preserved unchanged and remains visible as a model estimate.
    decision = "VERIFY" if not quality.is_useful else model_decision
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


# ============================================================
# HTML BUILDERS
# ============================================================

def hero_html(status: str, runtime: str | None = None) -> str:
    labels = {
        "online": ("LOCAL MODEL • ONLINE", ""),
        "loading": ("LOCAL MODEL • LOADING", "loading"),
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
                            <stop offset="0" stop-color="#4f46e5"/>
                            <stop offset="1" stop-color="#06b6d4"/>
                        </linearGradient>
                    </defs>
                    <path d="M12 2.5l8 3v6.2c0 4.6-3.2 8.3-8 9.8-4.8-1.5-8-5.2-8-9.8V5.5l8-3z" fill="url(#lg)"/>
                    <path d="M8.2 12.2l2.6 2.6 5-5.4" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
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
    icon = "⚠️" if kind == "warn" else "⛔"
    return f"""
    <div class="notice {kind}">
        <div>{icon}</div>
        <div><b>{esc(title)}</b><span>{esc(body)}</span></div>
    </div>
    """


def loading_html() -> str:
    return """
    <div class="loading card">
        <div class="spin"></div>
        <div>
            <div class="ltitle">Running the reliability gate</div>
            <div class="lsub">Inference runs locally on this machine — no cloud API involved.</div>
            <ul class="lsteps">
                <li style="--d:0s">Generating independent answer samples…</li>
                <li style="--d:0.4s">Comparing agreement…</li>
                <li style="--d:0.8s">Measuring uncertainty…</li>
                <li style="--d:1.2s">Running reliability gate…</li>
            </ul>
        </div>
        <div class="lbar"></div>
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


def gauge_svg(probability: float, threshold: float, decision: str) -> str:
    cx = cy = 140
    radius = 100
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - probability)

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

    if decision == "TRUST":
        stop_a, stop_b, word_color = "#34d399", "#06b6d4", "#059669"
    else:
        stop_a, stop_b, word_color = "#fcd34d", "#f97316", "#b45309"

    return f"""
    <svg class="gauge" viewBox="0 0 280 280" role="img"
         aria-label="Estimated reliability {probability * 100:.1f} percent, trust boundary {pct_plain(threshold)} percent">
        <defs>
            <linearGradient id="ringgrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0" stop-color="{stop_a}"/>
                <stop offset="1" stop-color="{stop_b}"/>
            </linearGradient>
        </defs>
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="#ffffff" stroke="#e8ecfb" stroke-width="18"/>
        <circle cx="{cx}" cy="{cy}" r="{zone_radius}" fill="none" stroke="#10b981" stroke-opacity="0.4"
                stroke-width="5" stroke-linecap="round" stroke-dasharray="{zone_dash:.2f} {zone_circ:.2f}"
                stroke-dashoffset="{-zone_start:.2f}" transform="rotate(-90 {cx} {cy})"/>
        <circle class="ring-prog" cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="url(#ringgrad)"
                stroke-width="18" stroke-linecap="round" stroke-dasharray="{circumference:.2f}"
                stroke-dashoffset="{offset:.2f}" style="--circ:{circumference:.2f};--off:{offset:.2f}"
                transform="rotate(-90 {cx} {cy})"/>
        <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#0f172a" stroke-width="3" stroke-linecap="round"/>
        <text x="{lx:.2f}" y="{ly + 4:.2f}" text-anchor="middle" font-size="12" font-weight="700" fill="#475569"
              style="font-family:inherit">{pct_plain(threshold)}%</text>
        <text x="{cx}" y="{cy + 14}" text-anchor="middle" font-size="50" font-weight="800" fill="#0f172a"
              style="font-family:inherit">{probability * 100:.1f}%</text>
        <text x="{cx}" y="{cy + 40}" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="#64748b"
              style="font-family:inherit">EST. RELIABILITY</text>
        <text x="{cx}" y="{cy + 66}" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="2" fill="{word_color}"
              style="font-family:inherit">{decision}</text>
    </svg>
    """


def result_html(view: dict) -> str:
    decision = view["decision"]
    state = decision.lower()
    probability = view["probability"]
    threshold = view["threshold"]
    thr_text = pct_plain(threshold)
    comparator = "&ge;" if probability >= threshold else "&lt;"

    if decision == "TRUST":
        icon = (
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            '<path d="M12 2.5l8 3v6.2c0 4.6-3.2 8.3-8 9.8-4.8-1.5-8-5.2-8-9.8V5.5l8-3z" fill="rgba(255,255,255,.25)" stroke="#fff" stroke-width="1.6"/>'
            '<path d="M8.2 12.2l2.6 2.6 5-5.4" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        message = (
            "<b>Trust recommended.</b> The sampled answers and token confidence "
            "cleared the gate's calibrated threshold. This is still an estimate, "
            "not proof."
        )
    else:
        icon = (
            '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
            '<circle cx="11" cy="11" r="6.5" stroke="#451a03" stroke-width="2"/>'
            '<path d="M16 16l4.5 4.5" stroke="#451a03" stroke-width="2.2" stroke-linecap="round"/></svg>'
        )
        if not view["quality_ok"]:
            message = (
                "<b>Response-quality safeguard triggered.</b> "
                f"{esc(view['quality_reason'])} The final recommendation is VERIFY. "
                "The reliability estimate is preserved, but it is not enough on its own "
                "when the answer does not appear to answer the question."
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

    if view["quality_ok"]:
        comparison_caption = (
            f"{probability * 100:.1f}% {comparator} {thr_text}% → "
            f'<em class="{state}">{decision}</em>'
        )
    else:
        comparison_caption = (
            'Response-quality check → <em class="verify">VERIFY</em>'
        )

    return f"""
    <div class="rc-wrap {state}">
        <div class="rc">
            <div>
                <div class="eyebrow">Consensus answer</div>
                <div class="answer">{answer_text}</div>
                {raw_output_html}
                <div class="badge-row">
                    <div class="badge {state}">{icon}{decision}</div>
                    <div class="thr-chip">Trust threshold: {thr_text}%</div>
                </div>
                <div class="decision-msg">{message}</div>
                <div class="qecho"><b>Question</b> · {esc(view["question"])}</div>
                <div class="disclaimer">
                    Reliability is the gate's calibrated estimate of the probability that
                    this answer is correct. It is a decision policy, not a guarantee of factual truth.
                </div>
            </div>
            <div class="gauge-col">
                <div class="eyebrow" style="margin-bottom:6px">Reliability estimate</div>
                {gauge_svg(probability, threshold, decision)}
                <div class="meter">
                    <div class="meter-fill {state}" style="--w:{probability * 100:.2f}%"></div>
                    <div class="meter-mark" style="left:{threshold * 100:.2f}%"><span>{thr_text}%</span></div>
                </div>
                <div class="meter-cap">
                    {comparison_caption}
                </div>
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
    <div class="mcard" style="--accent:{accent};animation-delay:{delay:.2f}s" title="{esc(note)}">
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
            "#6366f1",
            bar=view["agreement"],
            sub=f"{round(view['agreement'] * n)} of {n} samples in the largest group",
            delay=0.0,
        ),
        metric_card(
            "Consensus margin",
            f"{view['margin'] * 100:.0f}%",
            METRIC_NOTES["margin"],
            "#7c3aed",
            bar=view["margin"],
            sub="Winner vs. runner-up",
            delay=0.06,
        ),
        metric_card(
            "Answer-key entropy",
            f"{view['entropy']:.3f}",
            METRIC_NOTES["entropy"],
            "#0ea5e9",
            bar=view["entropy"] / max_entropy,
            sub=f"Maximum {max_entropy:.2f} for {n} distinct answers",
            delay=0.12,
        ),
        metric_card(
            "Mean token log-probability",
            f"{view['mean_lp']:.3f}",
            METRIC_NOTES["mean_lp"],
            "#06b6d4",
            sub=f"≈ {token_probability(view['mean_lp']) * 100:.0f}% geometric-mean token probability",
            delay=0.18,
        ),
        metric_card(
            "Minimum token log-probability",
            f"{view['min_lp']:.3f}",
            METRIC_NOTES["min_lp"],
            "#14b8a6",
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
        ("Reliability model", "Calibrated ExtraTrees"),
        ("Final decision", view["decision"]),
        ("Classifier decision", view["model_decision"]),
        ("Response quality", "Passed" if view["quality_ok"] else "Flagged: " + view["quality_reason"]),
        ("Runtime", view["runtime"]),
        ("Raw P(correct)", f"{view['probability']:.4f}"),
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


def sidebar_html(threshold: float, history: list[dict]) -> str:
    thr_text = pct_plain(threshold)

    history_html = ""
    if history:
        rows = "".join(
            f"""
            <div class="hist-item">
                <i class="{item['decision'].lower()}"></i>
                <em title="{esc(item['question'])}">{esc(item['question'])}</em>
                <b>{item['probability'] * 100:.0f}%</b>
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
    <div class="side-kv"><span>Threshold</span><b>{thr_text}%</b></div>
    <div class="policy">
        <small>Decision policy</small>
        P(correct) &ge; {thr_text}% → <span class="t">TRUST</span><br>
        P(correct) &lt; {thr_text}% → <span class="v">VERIFY</span>
    </div>
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
st.session_state.setdefault("history", [])


def use_example(example: str) -> None:
    st.session_state["question_input"] = example


def clear_all() -> None:
    st.session_state["question_input"] = ""
    st.session_state["result"] = None


# ============================================================
# PAGE
# ============================================================

render(CSS)
render(ORBS)

hero_slot = st.empty()
hero_slot.markdown(html_block(hero_html("loading")), unsafe_allow_html=True)

gate, load_error = try_load_gate()

if gate is not None:
    hero_slot.markdown(
        html_block(hero_html("online", runtime_label(gate))),
        unsafe_allow_html=True,
    )
else:
    hero_slot.markdown(html_block(hero_html("offline")), unsafe_allow_html=True)


# ---------------- input card ----------------

try:
    input_box = st.container(key="input_card")
except TypeError:  # older Streamlit without container keys
    input_box = st.container(border=True)

with input_box:
    render(
        """
        <div class="card-head">Ask the model</div>
        <div class="card-sub">
            Ask a factual question. The gate samples several answers and estimates
            whether you can rely on them.
        </div>
        """
    )

    st.text_area(
        "Question",
        key="question_input",
        placeholder="Ask a factual question...",
        height=110,
        label_visibility="collapsed",
    )

    render('<div class="mini-label">Try an example</div>')

    chip_columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, example in zip(chip_columns, EXAMPLE_QUESTIONS):
        with column:
            st.button(
                example,
                key=f"example_{example}",
                on_click=use_example,
                args=(example,),
                use_container_width=True,
            )

    cta_column, clear_column = st.columns([4, 1])
    with cta_column:
        run_clicked = st.button(
            "Check Reliability",
            type="primary",
            use_container_width=True,
        )
    with clear_column:
        st.button("Clear", on_click=clear_all, use_container_width=True)


# ---------------- run the gate (only on the click rerun) ----------------

notice_slot = st.empty()
status_slot = st.empty()

if run_clicked:
    question_text = st.session_state.get("question_input", "").strip()

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

    elif gate is None:
        notice_slot.markdown(
            html_block(
                notice_html(
                    "error",
                    (load_error or {}).get("title", "Model unavailable"),
                    (load_error or {}).get(
                        "body", "The local model is not available."
                    ),
                )
            ),
            unsafe_allow_html=True,
        )

    else:
        status_slot.markdown(html_block(loading_html()), unsafe_allow_html=True)

        try:
            new_view = run_check(gate, question_text)

        except (AttributeError, TypeError, ValueError, KeyError) as exc:
            logger.exception("Gate returned an unexpected result")
            st.session_state["result"] = None
            notice_slot.markdown(
                html_block(
                    notice_html(
                        "error",
                        "Unexpected result",
                        "The gate returned a result the interface could not read. "
                        "Please try again.",
                    )
                ),
                unsafe_allow_html=True,
            )

        except Exception as exc:
            logger.exception("Inference failed")
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

            notice_slot.markdown(
                html_block(notice_html("error", title, body)),
                unsafe_allow_html=True,
            )

        else:
            st.session_state["result"] = new_view
            st.session_state["history"] = (
                [
                    {
                        "question": new_view["question"],
                        "decision": new_view["decision"],
                        "probability": new_view["probability"],
                    }
                ]
                + st.session_state["history"]
            )[:HISTORY_LIMIT]

        finally:
            status_slot.empty()


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
    render(result_html(current))

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


# ---------------- sidebar (rendered last so session history is current) ----------------

with st.sidebar:
    render(
        sidebar_html(
            gate.threshold if gate is not None else DEFAULT_THRESHOLD,
            st.session_state["history"],
        )
    )