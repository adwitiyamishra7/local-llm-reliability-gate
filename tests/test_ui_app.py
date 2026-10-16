"""Streamlit AppTest coverage for the UI submission and lazy-load path."""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_streamlit_form_submission_updates_status_and_runs_once(monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    testing = pytest.importorskip("streamlit.testing.v1")
    from streamlit.delta_generator import DeltaGenerator

    import src.gate

    metrics = {"constructed": 0, "predicted": 0, "sampled": 0, "verified": 0}
    backend_ref = {}
    gate_instances = []

    class FakeClassifier:
        def predict_proba(self, _features):
            return [[0.01, 0.99]]

    class FakeGate:
        def __init__(self):
            metrics["constructed"] += 1
            self.calibrated = False
            self.calibration_note = (
                "The artifact fingerprint does not match the current configuration."
            )
            self.threshold = 0.82
            self.llm = SimpleNamespace(device=SimpleNamespace(type="cpu"))
            self.model = FakeClassifier()
            gate_instances.append(self)

        @staticmethod
        def _build_features(question, results):
            return {"question": question, "count": len(results)}

        @staticmethod
        def _select_consensus(answers, question):
            return answers[0], {"quality_ok": True}

        def predict(self, question, return_all_answers=True):
            metrics["predicted"] += 1
            backend = backend_ref["module"]
            samples = backend.sample_answers(self.llm, question)
            self._build_features(question=question, results=samples)
            self.model.predict_proba(None)
            answers = [sample.answer for sample in samples]
            answer, info = self._select_consensus(answers, question)
            if info["quality_ok"]:
                backend.verify_answer(question, answer)
            return SimpleNamespace(
                question=question,
                answer=answer,
                probability_correct=0.99,
                decision="TRUST",
                threshold=self.threshold,
                calibrated=self.calibrated,
                reliability_reported=True,
                fact_verification_status="VERIFIED",
                fact_verification_supported=True,
                fact_verification_reason="The fake test source verified the answer.",
                fact_verification_evidence="Fixture evidence.",
                fact_verification_source_url="",
                material_disagreement=False,
                raw_answers=answers if return_all_answers else [],
                answer_agreement=1.0,
                consensus_margin=1.0,
                answer_key_entropy=0.0,
                mean_logprob_avg=-0.1,
                min_logprob_avg=-0.2,
            )

    backend = ModuleType("src.gate.reliability_gate")
    backend.ReliabilityGate = FakeGate

    def sample_answers(_llm, _question):
        metrics["sampled"] += 1
        return [SimpleNamespace(answer="William Shakespeare") for _ in range(5)]

    def verify_answer(_question, _answer):
        metrics["verified"] += 1
        return SimpleNamespace(status="VERIFIED")

    backend.sample_answers = sample_answers
    backend.verify_answer = verify_answer
    backend_ref["module"] = backend
    monkeypatch.setitem(sys.modules, "src.gate.reliability_gate", backend)
    monkeypatch.setattr(src.gate, "reliability_gate", backend, raising=False)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))

    stages: list[str] = []
    loading_markup: list[str] = []
    stylesheet_markup: list[str] = []
    original_markdown = DeltaGenerator.markdown
    original_html = streamlit.html

    def record_loading(self, body, *args, **kwargs):
        if isinstance(body, str) and 'class="loading card"' in body:
            loading_markup.append(body)
            match = re.search(r'<div class="ltitle">(.*?)</div>', body)
            if match:
                stages.append(html.unescape(match.group(1)))
        return original_markdown(self, body, *args, **kwargs)

    def record_stylesheet(body, *args, **kwargs):
        if isinstance(body, str) and "<style>" in body:
            stylesheet_markup.append(body)
        return original_html(body, *args, **kwargs)

    def assert_stylesheet_pairs():
        assert len(stylesheet_markup) % 2 == 0
        for index in range(0, len(stylesheet_markup), 2):
            pair = stylesheet_markup[index:index + 2]
            big = [
                body for body in pair
                if "@media (max-width: 640px)" in body and "--grad:" in body
            ]
            theme = [
                body for body in pair
                if ":root{" in body and "color-scheme:" in body
            ]
            assert len(big) == 1 and len(theme) == 1
            assert pair == [big[0], theme[0]]

    monkeypatch.setattr(DeltaGenerator, "markdown", record_loading)
    monkeypatch.setattr(streamlit, "html", record_stylesheet)
    streamlit.cache_resource.clear()

    try:
        app = testing.AppTest.from_file(str(APP))
        app.query_params["theme"] = "IvOrY"
        app.run(timeout=60)
        assert not app.exception
        assert_stylesheet_pairs()
        initial_markup = "\n".join(item.value for item in app.markdown)
        assert 'class="notice info"' in initial_markup
        assert "Model not loaded yet" in initial_markup
        assert "Calibration refresh required" not in initial_markup
        assert metrics["constructed"] == 0  # first render stays lazy
        assert len(stylesheet_markup) == 2
        assert "--accent:#86198f;" in stylesheet_markup[1]
        assert app.session_state["theme_choice"] == "Ivory"

        # Empty form submission takes the UI validation path and never constructs a gate.
        app.text_input[0].set_value("   ")
        app.button[0].click().run(timeout=60)
        assert_stylesheet_pairs()
        empty_markup = "\n".join(item.value for item in app.markdown)
        assert "Enter a question first" in empty_markup
        assert metrics["constructed"] == 0

        # A changed question followed by the form submit runs the same production
        # load_and_run/stage-hook/result path used by the Check Reliability control.
        app.text_input[0].set_value("Who wrote Hamlet?")
        app.button[0].click().run(timeout=60)
        assert_stylesheet_pairs()
        assert not app.exception
        assert metrics == {"constructed": 1, "predicted": 1, "sampled": 1, "verified": 1}
        assert backend.sample_answers is sample_answers
        assert backend.verify_answer is verify_answer
        assert "predict_proba" not in vars(gate_instances[0].model)
        assert "_build_features" not in vars(gate_instances[0])
        assert "_select_consensus" not in vars(gate_instances[0])
        assert stages == [
            "Loading local model: Qwen3-0.6B",
            "Sampling 5 answers",
            "Extracting reliability features",
            "Classifying answer reliability",
            "Running independent verification",
            "Finalizing the recommendation",
        ]
        assert all("Still working" in markup for markup in loading_markup)
        assert all("%" not in markup for markup in loading_markup)

        result_markup = "\n".join(item.value for item in app.markdown)
        assert "The artifact fingerprint does not match the current configuration." in result_markup
        assert 'class="badge verify" role="status" aria-label="Final decision: VERIFY"' in result_markup
        assert "Model not loaded yet" not in result_markup

        # Theme changes rerender colors while retaining the question and result.
        app.segmented_control[0].set_value("Midnight").run(timeout=60)
        assert not app.exception
        assert_stylesheet_pairs()
        assert "--bg:#080c18;" in stylesheet_markup[-1]
        assert app.query_params["theme"] == ["midnight"]
        assert app.text_input[0].value == "Who wrote Hamlet?"
        assert app.session_state["result"] is not None
        assert metrics["predicted"] == 1
        assert 'class="badge verify" role="status" aria-label="Final decision: VERIFY"' in "\n".join(
            item.value for item in app.markdown
        )

        # The form submit is consumed once; an ordinary rerun does not infer again.
        app.run(timeout=60)
        assert_stylesheet_pairs()
        assert metrics["predicted"] == 1

        # Clear still resets the form value/result, and the visible submit control
        # can be used again without loading a second gate instance.
        next(button for button in app.button if button.label == "Clear").click().run(timeout=60)
        assert_stylesheet_pairs()
        assert app.text_input[0].value == ""
        app.text_input[0].set_value("Who wrote Hamlet?")
        app.button[0].click().run(timeout=60)
        assert_stylesheet_pairs()
        assert not app.exception
        assert metrics["constructed"] == 1
        assert metrics["predicted"] == 2

        # An example chip fills the box AND runs the check in one click, exactly once.
        next(button for button in app.button if button.label == "What is 15% of 240?").click().run(timeout=60)
        assert_stylesheet_pairs()
        assert not app.exception
        assert app.text_input[0].value == "What is 15% of 240?"
        assert metrics["constructed"] == 1
        assert metrics["predicted"] == 3
        app.run(timeout=60)
        assert_stylesheet_pairs()
        assert metrics["predicted"] == 3
    finally:
        streamlit.cache_resource.clear()
