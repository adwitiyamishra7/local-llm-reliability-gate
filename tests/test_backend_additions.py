"""Regression coverage for additive backend and deployment support."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.gate.config as cfg
import src.gate.model as model_module
import src.gate.reliability_gate as rg


LEGACY_RESULT_KEYS = [
    "question", "answer", "probability_correct", "decision", "threshold",
    "answer_agreement", "consensus_margin", "answer_key_entropy",
    "mean_logprob_avg", "min_logprob_avg", "raw_answers", "calibrated",
    "calibration_note", "prompt_version", "quality_ok", "quality_reason",
    "reliability_reported", "useful_samples", "nonanswer_majority",
    "material_disagreement", "sample_useful", "fact_verification_status",
    "fact_verification_supported", "fact_verification_reason",
    "fact_verification_evidence", "fact_verification_source_url",
]


def make_gate():
    gate = object.__new__(rg.ReliabilityGate)
    gate.model = SimpleNamespace(
        predict_proba=lambda frame: np.array([[0.01, 0.99]], dtype=float)
    )
    gate.feature_names = list(cfg.EXPECTED_FEATURES)
    gate.threshold = 0.72
    gate.calibrated = True
    gate.calibration_note = ""
    gate.llm = object()
    gate._build_features = lambda **kwargs: {name: 0.0 for name in cfg.EXPECTED_FEATURES}
    return gate


def fake_result(answer="36"):
    return SimpleNamespace(
        answer=answer,
        mean_logprob=-0.2,
        min_logprob=-0.4,
        finished=True,
    )


def test_progress_stage_order_and_callback_errors_do_not_break_gate(monkeypatch):
    monkeypatch.setattr(rg, "generate_one", lambda **kwargs: fake_result())
    monkeypatch.setattr(
        rg,
        "verify_answer",
        lambda question, answer: rg.VerificationResult("VERIFIED", True, "checked"),
    )
    gate = make_gate()
    events = []

    result = gate.predict(
        "What is 15% of 240?",
        progress_callback=lambda *event: events.append(event),
    )

    assert [event[0] for event in events] == (
        ["sampling"] * cfg.N_SAMPLES + ["agreement", "verification", "decision"]
    )
    assert [(event[1], event[2]) for event in events[:cfg.N_SAMPLES]] == [
        (index, cfg.N_SAMPLES) for index in range(1, cfg.N_SAMPLES + 1)
    ]
    assert all(event[3] == "36" for event in events[:cfg.N_SAMPLES])
    assert result.question_kind == "factual"
    assert events[-1][3] == result.decision

    broken_callback_result = gate.predict(
        "What is 15% of 240?",
        progress_callback=lambda *event: (_ for _ in ()).throw(RuntimeError("UI closed")),
    )
    assert broken_callback_result.decision == result.decision
    assert broken_callback_result.probability_correct == result.probability_correct
    assert broken_callback_result.raw_answers == result.raw_answers


def test_should_cancel_raises_gate_cancelled_between_samples(monkeypatch):
    generated = []
    monkeypatch.setattr(
        rg,
        "generate_one",
        lambda **kwargs: generated.append(kwargs) or fake_result(),
    )
    gate = make_gate()

    with pytest.raises(rg.GateCancelled):
        gate.predict("What is 15% of 240?", should_cancel=lambda: True)

    assert len(generated) == 1


def test_question_metadata_is_additive_and_uses_existing_verifier_support(monkeypatch):
    results = [fake_result() for _ in range(cfg.N_SAMPLES)]
    # Keep compatibility with existing two-argument sample_answers patches/callers.
    monkeypatch.setattr(rg, "sample_answers", lambda llm, question: results)
    monkeypatch.setattr(
        rg,
        "verify_answer",
        lambda question, answer: rg.VerificationResult("UNSUPPORTED", False, "unsupported"),
    )

    payload = make_gate().predict("Who is Ada Lovelace?").to_dict()

    assert list(payload) == LEGACY_RESULT_KEYS + ["question_kind", "user_message"]
    assert payload["question"] == "Who is Ada Lovelace?"
    assert payload["answer"] == "36"
    assert payload["probability_correct"] == 0.99
    assert payload["decision"] == "VERIFY"
    assert payload["fact_verification_status"] == "UNSUPPORTED"
    assert payload["question_kind"] == "unsupported"
    assert payload["user_message"] == (
        "This looks like an open-ended question, so the gate can't score it reliably. "
        "Verify the answer yourself."
    )

    poor_answers = [fake_result("I don't know.") for _ in range(cfg.N_SAMPLES)]
    monkeypatch.setattr(rg, "sample_answers", lambda llm, question: poor_answers)
    factual_but_unscorable = make_gate().predict("What is 15% of 240?")
    assert factual_but_unscorable.question_kind == "factual"
    assert factual_but_unscorable.fact_verification_status == "UNVERIFIED"


def test_warm_up_loads_and_generates_once(monkeypatch):
    target = SimpleNamespace()
    loaded = []
    generated = []
    monkeypatch.setattr(model_module, "_WARMUP_MODEL", None)
    monkeypatch.setattr(model_module, "_WARMUP_DONE", False)
    monkeypatch.setattr(model_module, "load_local_llm", lambda: loaded.append(target) or target)

    from src.gate import generation

    monkeypatch.setattr(
        generation,
        "generate_one",
        lambda llm, question, **kwargs: generated.append((llm, question, kwargs)),
    )

    assert model_module.warm_up() is target
    assert model_module.warm_up() is target
    assert len(loaded) == 1
    assert len(generated) == 1
    assert generated[0][2]["max_new_tokens"] == 1


def test_cpu_fallback_uses_float32_and_device_label(monkeypatch):
    monkeypatch.setattr(model_module.torch.cuda, "is_available", lambda: False)

    class FakeTokenizer:
        pad_token_id = None
        eos_token_id = 2
        pad_token = None
        eos_token = "eos"

    class FakeModel:
        def eval(self):
            return self

    model_calls = []
    monkeypatch.setattr(
        model_module.AutoTokenizer,
        "from_pretrained",
        lambda name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        model_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, **kwargs: model_calls.append(kwargs) or FakeModel(),
    )

    llm = model_module.load_local_llm()

    assert llm.device.type == "cpu"
    assert model_calls[0]["dtype"] is torch.float32
    assert model_calls[0]["device_map"] == "cpu"
    assert model_module.get_device_label() == "CPU"
    assert model_module.get_device_label().strip()
