"""UI regression tests for the reliability gauge (no Streamlit, torch or model needed).

Loads only the pure rendering/decision functions from app.py via AST, with the
response-quality check and answer-key extractor replaced by small fakes, so the
tests are hermetic. They cover the screenshot failure: answer "I don't know"
with classifier probability 0.745 and threshold 0.72.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"
WANTED = {"esc", "pct_plain", "token_probability", "answer_groups", "build_view",
          "gauge_svg", "result_html", "technical_html", "sidebar_html", "metric_card"}


try:  # prefer the REAL response-quality implementation when the project is importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.gate.response_quality import assess_response_quality as REAL_QUALITY
except Exception:  # pragma: no cover
    REAL_QUALITY = None


def fake_quality(question, answer):
    bad = (not answer.strip()) or answer.lower().startswith("i don't know")
    return SimpleNamespace(is_useful=not bad, reason="Non-answer or refusal" if bad else "ok")


def load_ui():
    import html as html_mod
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    ns = {
        "math": math, "Counter": Counter, "datetime": datetime, "html": html_mod,
        "assess_response_quality": REAL_QUALITY or fake_quality,
        "extract_answer_key": lambda q, a: " ".join(a.lower().split()),
        "MODEL_NAME": "Qwen/Qwen3-0.6B", "N_SAMPLES": 5, "EXPECTED_FEATURES": ["f"] * 15,
        "METRIC_NOTES": {}, "annotations": __import__("__future__").annotations,
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(APP), "exec"), ns)
    return ns


def result(answer, probability=0.745, threshold=0.72, decision="TRUST", calibrated=True):
    return SimpleNamespace(
        question="Who is Narendra Modi?", answer=answer, probability_correct=probability,
        threshold=threshold, decision=decision, calibrated=calibrated,
        raw_answers=[answer] * 5, answer_agreement=1.0, consensus_margin=1.0,
        answer_key_entropy=0.0, mean_logprob_avg=-0.1, min_logprob_avg=-0.5,
    )


@pytest.fixture(scope="module")
def ui():
    return load_ui()


def gauge_text(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def test_screenshot_case_idk_with_745_never_shows_gauge_percentage(ui):
    view = ui["build_view"](result("I don't know.", 0.745), 1.0, "CPU")
    assert view["decision"] == "VERIFY"
    assert view["score_valid"] is False
    html = ui["result_html"](view)
    start = html.index("<svg class=\"gauge\"")
    svg = html[start:html.index("</svg>", start)]
    assert "74.5" not in svg and "74.5%" not in html
    assert ">N/A<" in svg and "NOT REPORTED" in svg
    assert "No useful answer produced" in html
    assert "reliability not reported for a non-answer" in html
    assert "--w:0.00%" in html          # progress meter is empty
    assert "74.5" not in gauge_text(html)


def test_quality_failure_forces_verify_even_above_threshold(ui):
    for prob in (0.72, 0.745, 0.99):
        view = ui["build_view"](result("", prob), 1.0, "CPU")
        assert view["decision"] == "VERIFY" and view["model_decision"] == "TRUST"


def test_empty_output_is_not_useful(ui):
    view = ui["build_view"](result("   "), 1.0, "CPU")
    assert view["quality_ok"] is False and "(empty output)" in ui["result_html"](view)


def test_stale_calibration_hides_score_and_forces_verify(ui):
    view = ui["build_view"](result("Narendra Modi is the Prime Minister of India.", 0.9, calibrated=False), 1.0, "CPU")
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    html = ui["result_html"](view)
    assert "90.0%" not in html and "unavailable or unvalidated" in html


def test_valid_answer_valid_calibration_shows_score_and_policy(ui):
    view = ui["build_view"](result("William Shakespeare", 0.90), 1.0, "CPU")
    assert view["decision"] == "TRUST" and view["score_valid"] is True
    assert "90.0%" in ui["result_html"](view)
    low = ui["build_view"](result("Canberra", 0.5, decision="VERIFY"), 1.0, "CPU")
    assert low["decision"] == "VERIFY" and low["score_valid"] is True


def test_history_shows_na_for_failed_response(ui):
    bad = {"question": "q", "decision": "VERIFY", "probability": 0.745, "score_valid": False}
    good = {"question": "q2", "decision": "TRUST", "probability": 0.9, "score_valid": True}
    side = ui["sidebar_html"](0.72, [bad, good])
    assert "<b>N/A</b>" in side and "<b>90%</b>" in side and "<b>74%</b>" not in side and "<b>75%</b>" not in side


def test_technical_panel_labels_raw_score_diagnostic(ui):
    view = ui["build_view"](result("I don't know."), 1.0, "CPU")
    tech = ui["technical_html"](view)
    assert "diagnostic only" in tech and "Reliability reported" in tech
