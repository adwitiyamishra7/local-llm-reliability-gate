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
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"
WANTED = {"esc", "pct_plain", "token_probability", "answer_groups", "build_view",
          "gauge_svg", "result_html", "technical_html", "sidebar_html", "metric_card",
          "notice_html", "calibration_banner_html", "loading_html", "is_open_ended",
          "stage_step", "acquire_run_lock", "run_with_stage_hooks", "theme_css"}

THEME_TOKENS = (
    "--bg", "--surface", "--surface-2", "--border", "--text", "--muted",
    "--accent", "--accent-2", "--on-accent", "--trust", "--verify", "--error", "--info",
    "--trust-bg", "--trust-bd", "--verify-bg", "--verify-bd", "--error-bg", "--error-bd",
    "--info-bg", "--info-bd", "--track", "--shadow", "--hairline", "--glow-1", "--glow-2",
    "--code-bg",
)


try:  # prefer the REAL response-quality implementation when the project is importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.gate.response_quality import assess_response_quality as REAL_QUALITY
    from src.gate.agreement import answers_are_equivalent as ANSWERS_EQUIVALENT
except Exception:  # pragma: no cover
    REAL_QUALITY = None
    ANSWERS_EQUIVALENT = lambda *args, **kwargs: False


def fake_quality(question, answer):
    bad = (not answer.strip()) or answer.lower().startswith("i don't know")
    return SimpleNamespace(is_useful=not bad, reason="Non-answer or refusal" if bad else "ok")


def load_ui():
    import html as html_mod
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    themes = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "THEMES" for target in node.targets)
    )
    ns = {
        "math": math, "Counter": Counter, "datetime": datetime, "html": html_mod,
        "assess_response_quality": REAL_QUALITY or fake_quality,
        "answers_are_equivalent": ANSWERS_EQUIVALENT,
        "extract_answer_key": lambda q, a: " ".join(a.lower().split()),
        "MODEL_NAME": "Qwen/Qwen3-0.6B", "N_SAMPLES": 5, "EXPECTED_FEATURES": ["f"] * 15,
        "METRIC_NOTES": {}, "annotations": __import__("__future__").annotations,
        "re": re, "perf_counter": time.perf_counter,
        "logger": __import__("logging").getLogger("test_ui"),
        "run_check": lambda gate, question: gate.predict(question),
        "THEMES": themes,
    }
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(APP), "exec"), ns)
    return ns


def load_css() -> str:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CSS" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("CSS definition not found")


def result(answer, probability=0.745, threshold=0.72, decision="TRUST", calibrated=True,
           reliability_reported=True):
    return SimpleNamespace(
        question="Who is Narendra Modi?", answer=answer, probability_correct=probability,
        threshold=threshold, decision=decision, calibrated=calibrated,
        reliability_reported=reliability_reported,
        fact_verification_status="VERIFIED",
        fact_verification_supported=True,
        fact_verification_reason="Mocked independent verification passed.",
        fact_verification_evidence="Mock source matched the claim.",
        fact_verification_source_url="https://example.test/reference",
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


def test_missing_score_reporting_flag_hides_gauge_and_forces_verify(ui):
    value = result("William Shakespeare", 0.9)
    delattr(value, "reliability_reported")
    view = ui["build_view"](value, 1.0, "CPU")
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    assert "90.0%" not in ui["result_html"](view)


def test_missing_independent_verification_fails_closed(ui):
    value = result("William Shakespeare", 0.9)
    delattr(value, "fact_verification_status")
    view = ui["build_view"](value, 1.0, "CPU")
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    assert "90.0%" not in ui["result_html"](view)


def test_unreliable_verified_status_without_support_fails_closed(ui):
    value = result("William Shakespeare", 0.9)
    value.fact_verification_supported = False
    view = ui["build_view"](value, 1.0, "CPU")
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    assert "90.0%" not in ui["result_html"](view)


def test_material_disagreement_forces_verify_and_hides_score(ui):
    value = result("Sydney", 0.99)
    value.material_disagreement = True
    view = ui["build_view"](value, 1.0, "CPU")
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    html = ui["result_html"](view)
    assert "99.0%" not in html
    assert "Sample disagreement detected" in html


def test_valid_answer_valid_calibration_shows_score_and_policy(ui):
    view = ui["build_view"](result("William Shakespeare", 0.90), 1.0, "CPU")
    assert view["decision"] == "TRUST" and view["score_valid"] is True
    assert "90.0%" in ui["result_html"](view)
    assert "Independent verification: VERIFIED" in ui["result_html"](view)
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


def test_calibration_banner_distinguishes_preload_from_invalid_loaded_state(ui):
    preload = ui["calibration_banner_html"](False, False, "not checked yet")
    invalid_note = "The artifact fingerprint does not match the current configuration."
    invalid_loaded = ui["calibration_banner_html"](True, False, invalid_note)
    valid_loaded = ui["calibration_banner_html"](True, True, "")

    assert 'class="notice info"' in preload
    assert "Model not loaded yet" in preload
    assert "loads automatically" in preload
    assert "Calibration refresh required" not in preload
    assert 'class="notice warn"' in invalid_loaded
    assert "Calibration refresh required" in invalid_loaded
    assert invalid_note in invalid_loaded
    assert valid_loaded == ""


def test_trust_and_verify_badges_are_accessible_status_labels(ui):
    trust_view = ui["build_view"](result("William Shakespeare", 0.9), 1.0, "CPU")
    trust_html = ui["result_html"](trust_view)

    disagreement = result("Sydney", 0.99)
    disagreement.material_disagreement = True
    verify_view = ui["build_view"](disagreement, 1.0, "CPU")
    verify_html = ui["result_html"](verify_view)

    assert 'class="badge trust" role="status" aria-label="Final decision: TRUST"' in trust_html
    assert 'class="badge verify" role="status" aria-label="Final decision: VERIFY"' in verify_html
    assert 'aria-hidden="true"' in trust_html and 'aria-hidden="true"' in verify_html
    assert '<circle cx="12" cy="12" r="9"' in verify_html
    assert 'M16 16l4.5 4.5' not in verify_html
    assert "Sample disagreement detected" in verify_html

    css = load_css()
    badge_rules = css[css.index(".badge {"):css.index(".thr-chip {")]
    assert "cursor: default" in badge_rules
    assert "box-shadow: none" in badge_rules
    assert ".badge:hover" not in badge_rules


def test_loading_markup_is_indeterminate_and_explicitly_ongoing(ui):
    markup = ui["loading_html"]("Sampling 5 answers", "Still working · local inference can take a while.")
    assert 'role="status" aria-live="polite"' in markup
    assert "Sampling 5 answers" in markup
    assert "Still working" in markup
    assert "%" not in markup


def test_status_dot_is_eight_pixels_without_changing_pill_height():
    css = load_css()
    dot_rule = re.search(r"\.pill \.dot\.loading \{([^}]+)\}", css)
    pill_rule = re.search(r"\.pill \{([^}]+)\}", css)
    assert dot_rule and "width: 8px" in dot_rule.group(1) and "height: 8px" in dot_rule.group(1)
    assert pill_rule and "min-height: 32px" in pill_rule.group(1)


def test_global_stylesheet_has_direct_injection_and_responsive_accessible_rules():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    direct_css_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "html"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "CSS"
    ]
    assert direct_css_calls, "The global CSS must be injected as HTML, not Markdown."
    assert len(direct_css_calls) == 1
    theme_css_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "html"
        and node.args
        and isinstance(node.args[0], ast.Call)
        and isinstance(node.args[0].func, ast.Name)
        and node.args[0].func.id == "theme_css"
    ]
    assert len(theme_css_calls) == 1

    css = load_css()
    assert '[data-testid="stFormSubmitButton"] button' in css
    assert ":focus-visible" in css
    assert "@media (max-width: 640px)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ".rc-wrap.verify" in css and ".verification-note.verified" in css


@pytest.mark.parametrize("theme_name", ["daylight", "ivory", "midnight"])
def test_theme_contrast(ui, theme_name):
    theme = ui["THEMES"][theme_name]

    def color(token):
        value = theme[token]
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), f"{token} must be a hex color"
        return tuple(int(value[index:index + 2], 16) / 255 for index in (1, 3, 5))

    def luminance(rgb):
        linear = tuple(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in rgb)
        return sum(weight * channel for weight, channel in zip((0.2126, 0.7152, 0.0722), linear))

    def contrast(foreground, background):
        values = sorted((luminance(color(foreground)), luminance(color(background))), reverse=True)
        return (values[0] + 0.05) / (values[1] + 0.05)

    minimums = {
        ("--text", "--surface"): 7,
        ("--text", "--bg"): 7,
        ("--muted", "--surface"): 4.5,
        ("--muted", "--surface-2"): 4.5,
        ("--accent", "--surface"): 4.5,
        ("--on-accent", "--accent"): 4.5,
        ("--on-accent", "--accent-2"): 4.5,
        ("--trust", "--surface"): 4.5,
        ("--verify", "--surface"): 4.5,
        ("--error", "--surface"): 4.5,
        ("--info", "--surface"): 4.5,
    }
    for pair, minimum in minimums.items():
        assert contrast(*pair) >= minimum, f"{theme_name}: insufficient contrast for {pair}"


def test_valid_gauge_arc_uses_theme_accent_tokens(ui):
    markup = ui["gauge_svg"](0.9, 0.72, "TRUST")
    assert 'stop-color="var(--accent)"' in markup
    assert 'stop-color="var(--accent-2)"' in markup


def test_theme_css_has_no_angle_brackets(ui):
    for stylesheet in [load_css()] + [ui["theme_css"](name) for name in ui["THEMES"]]:
        body = stylesheet.replace("<style>", "").replace("</style>", "")
        assert "<" not in body
        assert "@property" not in body


def test_themes_define_every_token(ui):
    for name, theme in ui["THEMES"].items():
        assert all(token in theme for token in THEME_TOKENS), name
        block = ui["theme_css"](name)
        assert all(f"{token}:" in block for token in THEME_TOKENS), name
        assert f"color-scheme:{theme['color-scheme']}" in block


def test_no_hardcoded_colours_outside_theme_tokens():
    source = APP.read_text(encoding="utf-8")
    tree = ast.parse(source)
    themes_node = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "THEMES" for target in node.targets)
    )
    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def node_bounds(node):
        return (
            line_offsets[node.lineno - 1] + node.col_offset,
            line_offsets[node.end_lineno - 1] + node.end_col_offset,
        )

    themes_bounds = node_bounds(themes_node)
    css_start = source.index("CSS =")
    root_start = source.index(":root {", css_start)
    root_end = source.index("\n}", root_start) + 2
    allowed_ranges = [themes_bounds, (root_start, root_end)]
    color_pattern = re.compile(
        r"(?i)(?<![\w])#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})(?![\w])"
        r"|\b(?:rgba?|hsla?)\s*\([^)]*\)"
        r"|(?<![\w-])(?:white|black)(?![\w-])"
    )
    leftovers = [
        (source.count("\n", 0, match.start()) + 1, match.group(0))
        for match in color_pattern.finditer(source)
        if not any(start <= match.start() < end for start, end in allowed_ranges)
    ]
    assert leftovers == [], f"Hard-coded color literals outside theme tokens: {leftovers}"


# ---------------------------------------------------------------- V6 layout / live loading / lock


def test_unsupported_question_explains_na_instead_of_looking_broken(ui):
    value = result("The lotus is a plant known for large flowers.", 0.9)
    value.question = "Explain me lotus flower in 50 words"
    value.fact_verification_status = "UNSUPPORTED"
    value.fact_verification_supported = False
    view = ui["build_view"](value, 1.0, "CPU")
    html = ui["result_html"](view)
    assert view["decision"] == "VERIFY" and view["score_valid"] is False
    assert "Can&#x27;t be independently verified" in html
    assert "Only percentage math and capital cities have a verifier" in html
    assert "does not mean the answer is wrong." in html
    assert "This question type can't be scored" not in html
    assert "No verifier for this question type" not in html
    assert 'class="thr-chip"' not in html
    assert 'class="decision-msg"' not in html
    assert 'class="disclaimer"' not in html
    assert "Open-ended request" in html      # the 50-words hint
    assert "90.0%" not in html and ">N/A<" in html


def test_verify_search_link_and_fresh_answer_reveal(ui):
    value = result("William Shakespeare", 0.9)
    value.question = "Who is 5 & 7?"
    value.decision = "VERIFY"
    view = ui["build_view"](value, 1.0, "CPU")
    fresh_html = ui["result_html"](view, fresh=True)

    assert 'class="verify-link" href="https://www.google.com/search?q=Who+is+5+%26+7%3F"' in fresh_html
    assert 'target="_blank" rel="noopener noreferrer">Search the web for this</a>' in fresh_html
    assert fresh_html.count('class="answer-word"') == 2
    assert 'class="ring-prog fresh-ring"' in fresh_html
    assert 'class="rc-wrap verify fresh"' in fresh_html

    rerun_html = ui["result_html"](view)
    assert 'class="answer-word"' not in rerun_html
    assert 'class="ring-prog fresh-ring"' not in rerun_html

    long_view = dict(view)
    long_view["answer"] = " ".join(["word"] * 65)
    assert ui["result_html"](long_view, fresh=True).count('class="answer-word"') == 60

    unavailable_view = dict(view, score_valid=False)
    unavailable_view["fact_verification_status"] = "UNVERIFIED"
    unavailable_view["fact_verification_supported"] = False
    unavailable_view["decision"] = "VERIFY"
    unavailable_html = ui["result_html"](unavailable_view, fresh=True)
    assert ">N/A<" in unavailable_html
    assert 'class="ring-prog fresh-ring"' not in unavailable_html

    value.decision = "TRUST"
    trust_html = ui["result_html"](ui["build_view"](value, 1.0, "CPU"))
    assert "Search the web for this" not in trust_html


def test_open_ended_detection_is_a_hint_only(ui):
    check = ui["is_open_ended"]
    assert check("Explain me lotus flower in 50 words")
    assert check("Describe photosynthesis")
    assert not check("What is the capital of Australia?")
    assert not check("Who wrote Hamlet?")
    assert not check("What is 15% of 240?")
    plain = ui["result_html"](ui["build_view"](result("William Shakespeare", 0.9), 1.0, "CPU"))
    assert "Open-ended request" not in plain


def test_result_shows_latency_and_runtime(ui):
    html = ui["result_html"](ui["build_view"](result("William Shakespeare", 0.9), 12.34, "CUDA · GTX"))
    assert "Checked in 12.3 s" in html and "CUDA · GTX" in html


def test_stage_step_mapping(ui):
    step = ui["stage_step"]
    assert step("Loading local model: Qwen3-0.6B") == 0
    assert step("Sampling 5 answers") == 1
    assert step("Extracting reliability features") == 2
    assert step("Classifying answer reliability") == 2
    assert step("Running independent verification") == 3
    assert step("Finalizing the recommendation") == 4


def test_loading_pipeline_shows_real_progress_and_escapes_text(ui):
    markup = ui["loading_html"](
        "Sampling 5 answers", "Still working · 2 of 5 answers written.",
        active=1, samples_done=2, samples_total=5,
        previews=["Canberra", "<script>alert(1)</script> $5"], elapsed=7.9, model_ready=True,
    )
    assert 'class="loading card"' in markup and 'role="status" aria-live="polite"' in markup
    assert '<div class="ltitle">Sampling 5 answers</div>' in markup
    assert markup.count('class="pip done"') == 2 and markup.count('class="pip active"') == 1
    assert "2 of 5 answers written" in markup
    assert "<script>" not in markup and "&lt;script&gt;" in markup
    assert '<li class="done">' in markup and '<li class="active">' in markup and '<li class="todo">' in markup
    assert ">7 s</span>" in markup
    assert '<div class="pv new"><b>02</b>' in markup and '<div class="pv"><b>01</b>' in markup
    assert "CONSENSUS ANSWER" in markup and "PREPARING" in markup and "MEASURING" in markup
    assert "Thinking" in markup and "Candidate answers" in markup
    assert markup.count('class="candidate new"') == 1
    assert "%" not in markup

    percent_input = ui["loading_html"](
        "Sampling 5 answers 90%", "Still working · 50% done.",
        previews=["The result is 100%"],
    )
    assert "%" not in percent_input


def test_hooks_report_every_sample_and_restore_everything(ui, monkeypatch):
    import src.gate

    module = ModuleType("src.gate.reliability_gate")
    exec(
        "_calls = {'n': 0}\n"
        "def generate_one(llm, question, **kwargs):\n"
        "    _calls['n'] += 1\n"
        "    return type('R', (), {'answer': 'a%d' % _calls['n']})()\n"
        "def sample_answers(llm, question):\n"
        "    return [generate_one(llm, question) for _ in range(5)]\n"
        "def verify_answer(question, answer):\n"
        "    return 'checked'\n",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, "src.gate.reliability_gate", module)
    monkeypatch.setattr(src.gate, "reliability_gate", module, raising=False)
    original_sample, original_generate, original_verify = (
        module.sample_answers, module.generate_one, module.verify_answer,
    )

    def build_features(question, results):
        return {}

    def select_consensus(answers, question):
        return answers[0], {"quality_ok": True}

    def predict_proba(frame):
        return [[0.1, 0.9]]

    def predict(question):
        samples = module.sample_answers(None, question)
        gate._build_features(question, samples)
        gate.model.predict_proba(None)
        answer, _info = gate._select_consensus([s.answer for s in samples], question)
        module.verify_answer(question, answer)
        return "done"

    gate = SimpleNamespace(
        model=SimpleNamespace(predict_proba=predict_proba),
        _build_features=build_features,
        _select_consensus=select_consensus,
        predict=predict,
    )
    stages, progress = [], []
    outcome = ui["run_with_stage_hooks"](
        gate, "q", stages.append, lambda done, total, text: progress.append((done, total, text))
    )
    assert outcome == "done"
    assert progress == [(1, 5, "a1"), (2, 5, "a2"), (3, 5, "a3"), (4, 5, "a4"), (5, 5, "a5")]
    assert stages[0] == "Sampling 5 answers"
    assert stages[1:] == [
        "Extracting reliability features", "Classifying answer reliability",
        "Running independent verification", "Finalizing the recommendation",
    ]
    assert module.sample_answers is original_sample
    assert module.generate_one is original_generate
    assert module.verify_answer is original_verify
    assert gate._build_features is build_features and gate._select_consensus is select_consensus
    assert gate.model.predict_proba is predict_proba


def test_progress_callback_failure_never_breaks_inference(ui, monkeypatch):
    import src.gate

    module = ModuleType("src.gate.reliability_gate")
    exec(
        "def generate_one(llm, question, **kwargs):\n"
        "    return type('R', (), {'answer': 'x'})()\n"
        "def sample_answers(llm, question):\n"
        "    return [generate_one(llm, question) for _ in range(5)]\n"
        "def verify_answer(question, answer):\n"
        "    return None\n",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, "src.gate.reliability_gate", module)
    monkeypatch.setattr(src.gate, "reliability_gate", module, raising=False)
    gate = SimpleNamespace(
        model=SimpleNamespace(predict_proba=lambda frame: None),
        _build_features=lambda q, r: None,
        _select_consensus=lambda a, q: (a[0], {"quality_ok": True}),
        predict=lambda q: len(module.sample_answers(None, q)),
    )

    def broken(done, total, text):
        raise RuntimeError("display problem")

    assert ui["run_with_stage_hooks"](gate, "q", lambda stage: None, broken) == 5


def test_run_lock_is_waited_for_not_rejected(ui):
    state = {"lock": threading.Lock()}
    state["lock"].acquire()
    waits = []
    threading.Timer(0.7, state["lock"].release).start()
    assert ui["acquire_run_lock"](state, waits.append, timeout=5.0) is True
    assert waits, "the wait callback should have been used while the lock was busy"
    state["lock"].release()

    held = {"lock": threading.Lock()}
    held["lock"].acquire()
    assert ui["acquire_run_lock"](held, lambda waited: None, timeout=0.5) is False


def test_lock_is_released_before_any_streamlit_call_in_the_finally_block():
    """Regression for 'A check is already running' after pressing Clear mid-run.

    A rerun makes Streamlit raise from the next st.* call. If status_slot.empty() ran first
    inside finally, the lock release below it was skipped and the lock leaked forever.
    """
    source = APP.read_text(encoding="utf-8")
    match = re.search(r"finally:\n((?:[ ]+.*\n){1,12}?)\s*\n\s*paint_sidebar\(\)", source)
    assert match, "run handler finally block not found"
    block = match.group(1)
    assert block.index('state["lock"].release()') < block.index("status_slot.empty()")


def test_deploy_button_is_hidden_in_css_and_config():
    css = load_css()
    assert "stAppDeployButton" in css and "display: none !important" in css
    config = APP.parent / ".streamlit" / "config.toml"
    if config.exists():
        text = config.read_text(encoding="utf-8")
        assert 'toolbarMode = "minimal"' in text
        assert 'base = "light"' in text
        assert 'primaryColor = "#2563eb"' in text
        assert 'backgroundColor = "#f4f7fd"' in text
        assert 'secondaryBackgroundColor = "#eaf0fb"' in text
        assert 'textColor = "#0b1a33"' in text


def test_css_contains_no_angle_brackets_that_the_html_sanitizer_would_choke_on():
    """st.html() sanitizes the stylesheet. A literal '<' (for example @property syntax: "<integer>")
    can make it drop the WHOLE style block and leave the page completely unstyled."""
    css = load_css()
    assert "<" not in css.replace("<style>", "").replace("</style>", "")
    assert "@property" not in css
