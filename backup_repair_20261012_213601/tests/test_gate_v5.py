"""Mocked unit tests for the V5 generation/consensus/decision/calibration logic.
No model download, GPU or Internet is needed. Real-model checks: scripts/smoke_test_real_model.py
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.gate.config as cfg
import src.gate.generation as gen
import src.gate.reliability_gate as rg
from src.gate.response_quality import assess_response_quality as quality, clean_generated_text


# ---------------------------------------------------------------- response quality
@pytest.mark.parametrize("q,a", [
    ("Who is Narendra Modi?", "Narendra Modi is the Prime Minister of India and a leader of the BJP."),
    ("Who is Isaac Newton?", "Isaac Newton was an English physicist and mathematician known for the laws of motion and gravity."),
    ("What is the capital of Australia?", "Canberra"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("What is 15% of 240?", "36"),
    ("Who was Jack the Ripper?", "Jack the Ripper was an unidentified serial killer in 1888 London, never caught."),
    ("When was he born?", "Historians do not know his exact birth date, but it was around 1250."),
])
def test_valid_answers_pass(q, a):
    assert quality(q, a).is_useful, quality(q, a).reason


@pytest.mark.parametrize("q,a", [
    ("Who is Narendra Modi?", "I don't know."),
    ("Who is Isaac Newton?", "I'm sorry, but I don't have information about that person."),
    ("Who is Isaac Newton?", "I'm not familiar with that person."),
    ("Who is Isaac Newton?", "Unknown"),
    ("Who is Isaac Newton?", "Isaac Newton"),                      # echo of subject
    ("Who is Isaac Newton?", "Who is Isaac Newton?"),              # echoed question
    ("What is 15% of 240?", "240"),
    ("Who is Isaac Newton?", ""), ("Who is Isaac Newton?", "   "),  # empty
    ("Who is Isaac Newton?", "<think>\nLet me think"),             # hidden reasoning leaked
    ("Who is Isaac Newton?", "Isaac Newton was an English physicist who"),  # truncated
    ("Who is Isaac Newton?", "He wrote:"),
    ("Q", "x = $3"),                                               # unbalanced math
])
def test_failures_rejected(q, a):
    assert not quality(q, a).is_useful


def test_clean_generated_text():
    assert clean_generated_text("<think>\nabc\n</think>\n\nCanberra") == "Canberra"
    assert clean_generated_text("<think>never closed") == ""
    assert clean_generated_text("Canberra") == "Canberra"


# ---------------------------------------------------------------- config / calibration
def test_prompt_no_longer_quotes_refusal_phrase_but_allows_abstention():
    assert "I don't know" not in cfg.SYSTEM_PROMPT
    assert "not familiar" in cfg.SYSTEM_PROMPT and "Do not guess" in cfg.SYSTEM_PROMPT
    assert cfg.PROMPT_VERSION == "qa-useful-v5"


def test_fingerprint_changed_from_v4_and_tracks_settings(monkeypatch):
    assert cfg.config_fingerprint() != "6b9e2e2bde6ee939"
    before = cfg.config_fingerprint()
    monkeypatch.setattr(cfg, "TEMPERATURE", 0.9)
    assert cfg.config_fingerprint() != before


def v4_like_config():
    c = cfg.generation_config()
    for k in ("answer_cleaning", "quality_rule", "consensus_rule", "retry_policy"):
        c.pop(k)
    c.update(prompt_version="qa-useful-v4", temperature=0.8, top_p=0.9, top_k=50, prompt_sha256="0" * 64)
    return c


def test_v4_classifier_is_not_calibrated_for_v5():
    ok, note = rg.ReliabilityGate._check_calibration({"generation_config": v4_like_config()})
    assert not ok and "prompt_version" in note and "temperature" in note


def test_missing_config_is_not_calibrated_and_matching_config_is():
    assert not rg.ReliabilityGate._check_calibration({})[0]
    assert rg.ReliabilityGate._check_calibration({"generation_config": cfg.generation_config()})[0]


# ---------------------------------------------------------------- live == benchmark sampling
def test_sample_answers_uses_config_and_is_bounded(monkeypatch):
    calls = []

    def fake_generate_one(**kw):
        calls.append(kw)
        return SimpleNamespace(answer="x")

    monkeypatch.setattr(rg, "generate_one", fake_generate_one)
    out = rg.sample_answers(object(), "Q?")
    assert len(out) == len(calls) == cfg.N_SAMPLES            # no retry / resampling
    for kw in calls:
        assert (kw["max_new_tokens"], kw["do_sample"], kw["temperature"], kw["top_p"], kw["top_k"]) == (
            cfg.MAX_NEW_TOKENS, True, cfg.TEMPERATURE, cfg.TOP_P, cfg.TOP_K)


def test_build_prompt_uses_shared_system_prompt_and_disables_thinking():
    seen = {}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            seen.update(messages=messages, kw=kw)
            return "prompt"

    gen.build_prompt(Tok(), "Who is Isaac Newton?")
    assert seen["messages"][0] == {"role": "system", "content": cfg.SYSTEM_PROMPT}
    assert seen["messages"][1] == {"role": "user", "content": "Who is Isaac Newton?"}
    assert seen["kw"]["enable_thinking"] is False and seen["kw"]["add_generation_prompt"] is True


# ---------------------------------------------------------------- decisions
class FakeClassifier:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, frame):
        assert list(frame.columns) == cfg.EXPECTED_FEATURES and frame.shape == (1, 15)
        import numpy as np
        return np.array([[1 - self.p, self.p]])


def make_gate(prob, calibrated=True, threshold=0.72):
    g = object.__new__(rg.ReliabilityGate)
    g.model, g.feature_names, g.threshold = FakeClassifier(prob), list(cfg.EXPECTED_FEATURES), threshold
    g.calibrated, g.calibration_note, g.llm = calibrated, "" if calibrated else "stale", object()
    return g


def run(monkeypatch, gate, answers, q="Who is Isaac Newton?"):
    results = [SimpleNamespace(answer=a, mean_logprob=-0.2, min_logprob=-0.9, finished=True) for a in answers]
    monkeypatch.setattr(rg, "sample_answers", lambda llm, question: results)
    return gate.predict(q)


GOOD = "Isaac Newton was an English physicist and mathematician."


@pytest.mark.parametrize("p", [0.72, 0.745, 0.99])
def test_refusal_consensus_is_verify_even_above_threshold(monkeypatch, p):
    r = run(monkeypatch, make_gate(p), ["I don't know."] * 5)
    assert r.decision == "VERIFY" and not r.quality_ok and not r.reliability_reported
    assert r.raw_answers == ["I don't know."] * 5          # candidates preserved for diagnostics


def test_useful_answer_follows_threshold_policy(monkeypatch):
    assert run(monkeypatch, make_gate(0.9), [GOOD] * 5).decision == "TRUST"
    assert run(monkeypatch, make_gate(0.5), [GOOD] * 5).decision == "VERIFY"
    r = run(monkeypatch, make_gate(0.9), [GOOD] * 5)
    assert r.reliability_reported and r.quality_ok and r.useful_samples == 5


def test_stale_calibration_blocks_trust_and_score(monkeypatch):
    r = run(monkeypatch, make_gate(0.99, calibrated=False), [GOOD] * 5)
    assert r.decision == "VERIFY" and not r.reliability_reported and not r.calibrated


def test_refusals_cannot_win_consensus_over_useful_answers(monkeypatch):
    r = run(monkeypatch, make_gate(0.99), ["I don't know."] * 3 + [GOOD] * 2)
    assert r.answer == GOOD and r.nonanswer_majority and r.decision == "VERIFY"
    r2 = run(monkeypatch, make_gate(0.99), ["I don't know."] * 2 + [GOOD] * 3)
    assert r2.answer == GOOD and not r2.nonanswer_majority and r2.decision == "TRUST"


def test_useful_tie_goes_to_first_seen_not_longest():
    short, long_ = "Canberra", "Canberra is not it; Sydney is the largest city but the capital is Melbourne."
    ans, _ = rg.ReliabilityGate._select_consensus([short, long_], "What is the capital of Australia?")
    assert ans == short


def test_gate_result_feeds_ui_without_percentage(monkeypatch):
    from test_ui_regression import load_ui
    ui = load_ui()
    res = run(monkeypatch, make_gate(0.745), ["I don't know."] * 5, q="Who is Narendra Modi?")
    view = ui["build_view"](res, 1.0, "CPU")
    html = ui["result_html"](view)
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    assert "74.5" not in html and "N/A" in html
