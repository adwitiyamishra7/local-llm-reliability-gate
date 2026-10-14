"""Apply the V4 repair to the project. Run from the project root:

    python v4_repair\\apply_v4_patches.py

All-or-nothing: every anchor must match exactly once or NOTHING is written.
Originals are saved next to each file as <name>.bak_v3 (never overwritten).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path.cwd()


def rd(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def sub(t: str, old: str, new: str, label: str) -> str:
    n = t.count(old)
    if n != 1:
        raise SystemExit(f"[{label}] anchor matched {n} times (expected 1). Nothing was changed.")
    return t.replace(old, new)


def span(t: str, start: str, end: str, new: str, label: str) -> str:
    if t.count(start) != 1:
        raise SystemExit(f"[{label}] start anchor matched {t.count(start)} times. Nothing was changed.")
    i = t.index(start)
    j = t.find(end, i + len(start))
    if j < 0:
        raise SystemExit(f"[{label}] end anchor not found. Nothing was changed.")
    return t[:i] + new + t[j:]


out: dict[str, str] = {}

# ----------------------------------------------------------------- generation.py
p = "src/gate/generation.py"
t = rd(p)
t = sub(t, "from .model import LocalLLM\n",
        "from .config import MAX_NEW_TOKENS, SYSTEM_PROMPT\nfrom .model import LocalLLM\n", "gen-import")
t = sub(t, "    latency_seconds: float\n\n\ndef build_prompt",
        "    latency_seconds: float\n    finished: bool = True  # False if generation hit max_new_tokens without an end token\n\n\ndef build_prompt",
        "gen-dataclass")
t = span(t, '            "content": (\n                "You are a concise',
         '        },\n        {\n            "role": "user",',
         '            "content": SYSTEM_PROMPT,\n', "gen-prompt")
t = sub(t, "    max_new_tokens: int = 16,", "    max_new_tokens: int = MAX_NEW_TOKENS,", "gen-maxtok")
t = sub(t, "    return GenerationResult(\n        question=question,",
        '''    eos_ids = {llm.tokenizer.eos_token_id}
    configured_eos = getattr(llm.model.generation_config, "eos_token_id", None)
    if isinstance(configured_eos, int):
        eos_ids.add(configured_eos)
    elif configured_eos:
        eos_ids.update(configured_eos)
    finished = any(token_id in eos_ids for token_id in ids)

    return GenerationResult(
        question=question,''', "gen-finished")
t = sub(t, "        latency_seconds=latency,\n    )", "        latency_seconds=latency,\n        finished=finished,\n    )", "gen-return")
out[p] = t

# ----------------------------------------------------------------- reliability_gate.py
p = "src/gate/reliability_gate.py"
t = rd(p)
t = span(t, 'MODEL_NAME = "Qwen/Qwen3-0.6B"', "@dataclass\nclass GateResult:",
         '''from src.gate.config import (
    EXPECTED_FEATURES,
    GATE_FILE_LEGACY,
    GATE_FILE_V4,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    N_SAMPLES,
    PROMPT_VERSION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    config_differences,
    generation_config,
)

FINAL_GATE_FILE = GATE_FILE_V4


''', "rg-consts")
t = sub(t, "    raw_answers: list[str]\n\n    def to_dict",
        '''    raw_answers: list[str]

    calibrated: bool = True
    calibration_note: str = ""
    prompt_version: str = PROMPT_VERSION

    def to_dict''', "rg-result")
t = sub(t, '''        truncation_rate = (
            sum(_truncation_flag(answer) for answer in answers)
            / len(answers)
        )''', '''        # Truncated = suspicious word ending OR generation stopped without an end token.
        truncation_rate = (
            sum(
                1 if (_truncation_flag(answer) or not getattr(result, "finished", True)) else 0
                for answer, result in zip(answers, results)
            )
            / len(answers)
        )''', "rg-trunc")
t = sub(t, "class ReliabilityGate:", '''def sample_answers(llm, question: str) -> list[GenerationResult]:
    """The ONE sampling routine used by live inference and the benchmark runner."""
    return [
        generate_one(
            llm=llm,
            question=question,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        )
        for _ in range(N_SAMPLES)
    ]


class ReliabilityGate:''', "rg-sample")
t = span(t, "    def __init__(", "    @staticmethod\n    def _build_features(", '''    def __init__(
        self,
        gate_file: Path | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        if gate_file is None:
            gate_file = GATE_FILE_V4 if GATE_FILE_V4.exists() else GATE_FILE_LEGACY
        self.gate_file = Path(gate_file)

        if not self.gate_file.exists():
            raise FileNotFoundError(
                "Final reliability model not found:\\n"
                f"{self.gate_file}\\n\\n"
                "Run scripts\\\\finalize_gate_v4.py first."
            )

        payload = joblib.load(self.gate_file)

        if not isinstance(payload, dict):
            raise ValueError("Invalid final gate file: expected a dictionary payload.")

        missing = {"model", "feature_names", "trust_threshold"} - set(payload.keys())
        if missing:
            raise ValueError(
                "Final gate file is missing required fields: " + ", ".join(sorted(missing))
            )

        self.model = payload["model"]
        self.feature_names = list(payload["feature_names"])
        self.threshold = float(payload["trust_threshold"])
        self.model_name = model_name

        if self.feature_names != EXPECTED_FEATURES:
            raise ValueError(
                "Feature mismatch between training and live gate.\\n"
                f"Expected: {EXPECTED_FEATURES}\\n"
                f"Saved:    {self.feature_names}"
            )

        # Is this classifier calibrated for the CURRENT prompt/generation settings?
        self.calibrated, self.calibration_note = self._check_calibration(payload)

        # Load Qwen once. The model stays in memory for repeated questions.
        self.llm = load_local_llm(
            model_name=self.model_name,
            prefer_gpu=True,
        )

    @staticmethod
    def _check_calibration(payload: dict) -> tuple[bool, str]:
        saved = payload.get("generation_config")
        if not isinstance(saved, dict):
            return False, (
                "This gate was trained on benchmark answers from an earlier prompt and has no "
                "recorded generation settings. Its probabilities are not calibrated for the "
                f"current prompt ({PROMPT_VERSION}). Re-run the V4 recalibration workflow."
            )
        diffs = config_differences(saved, generation_config())
        if diffs:
            return False, (
                "Live prompt/generation settings differ from the ones this gate was trained "
                "with: " + "; ".join(diffs)
            )
        return True, ""

''', "rg-init")
t = span(t, "        results: list[GenerationResult] = []\n", "        features = self._build_features(",
         "        results = sample_answers(self.llm, question)\n\n", "rg-loop")
t = sub(t, '''        decision = (
            "TRUST"
            if probability >= self.threshold
            else "VERIFY"
        )''', '''        # An uncalibrated gate can never recommend TRUST.
        decision = (
            "TRUST"
            if (self.calibrated and probability >= self.threshold)
            else "VERIFY"
        )''', "rg-decision")
t = sub(t, "            raw_answers=answers,\n        )",
        "            raw_answers=answers,\n            calibrated=self.calibrated,\n            calibration_note=self.calibration_note,\n            prompt_version=PROMPT_VERSION,\n        )",
        "rg-return")
out[p] = t

# ----------------------------------------------------------------- risk_coverage_analysis.py
p = "scripts/risk_coverage_analysis.py"
t = rd(p)
t = sub(t, "from pathlib import Path\nimport sys\n", "import argparse\nimport json\nfrom pathlib import Path\nimport sys\n", "rc-imports")
t = sub(t, 'INPUT_CSV = ROOT_DIR / "artifacts" / "results" / "benchmark_results_300_v3.csv"\n',
        '''_ap = argparse.ArgumentParser()
_ap.add_argument("--input", default="benchmark_results_300_v3.csv")
_ap.add_argument("--tag", default="")  # e.g. v4 -> *_v4 outputs; empty keeps V3 file names
_args, _ = _ap.parse_known_args()
INPUT_CSV = ROOT_DIR / "artifacts" / "results" / _args.input
TAG = f"_{_args.tag}" if _args.tag else ""
''', "rc-input")
t = sub(t, "def select_threshold_from_oof(y_true: np.ndarray, oof_probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:",
        "def select_threshold_from_oof(y_true: np.ndarray, oof_probabilities: np.ndarray) -> tuple[float, pd.DataFrame, bool]:", "rc-sig")
t = sub(t, '        return float(selected["threshold"]), table\n', '        return float(selected["threshold"]), table, True\n', "rc-ret1")
t = sub(t, '    return float(fallback["threshold"]), table\n', '    return float(fallback["threshold"]), table, False\n', "rc-ret2")
t = sub(t, "    selected_threshold, oof_table = select_threshold_from_oof(", "    selected_threshold, oof_table, oof_target_met = select_threshold_from_oof(", "rc-call")
t = sub(t, '        "train_rows": len(X_train),\n', '        "train_rows": len(X_train),\n        "oof_target_met": bool(oof_target_met),\n', "rc-summary")
for name, expr in [
    ('"risk_coverage_thresholds.csv"', 'f"risk_coverage_thresholds{TAG}.csv"'),
    ('"risk_coverage_split_summaries.csv"', 'f"risk_coverage_split_summaries{TAG}.csv"'),
    ('"test_predictions_seed42.csv"', 'f"test_predictions_seed42{TAG}.csv"'),
    ('"risk_coverage_summary.txt"', 'f"risk_coverage_summary{TAG}.txt"'),
]:
    t = sub(t, f"OUTPUT_DIR / {name}", f"OUTPUT_DIR / {expr}", "rc-path-" + name)
t = sub(t, '    curve_path = OUTPUT_DIR / "risk_coverage_curve.png"\n', '''    curve_path = OUTPUT_DIR / f"risk_coverage_curve{TAG}.png"

    # Machine-readable deployment threshold (train-OOF only) for finalize_gate_v4.py
    (OUTPUT_DIR / f"selected_threshold{TAG}.json").write_text(
        json.dumps(
            {
                "threshold": float(robust_threshold),
                "selection_rule": selection_reason,
                "seeds": SEEDS,
                "per_seed_thresholds": [float(x) for x in split_summary_df["selected_threshold"]],
                "seeds_meeting_oof_target": int(split_summary_df["oof_target_met"].sum()),
                "min_trusted_accuracy_target": MIN_TRUSTED_ACCURACY,
                "input_file": INPUT_CSV.name,
                "rows": int(len(df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
''', "rc-json")
out[p] = t

# ----------------------------------------------------------------- validate_live_gate.py
p = "scripts/validate_live_gate.py"
t = rd(p)
t = sub(t, '"live_validation_10.csv"', '"live_validation_10_v4.csv"', "val-out")
t = sub(t, '                "consensus_answer": result.answer,\n',
        '                "consensus_answer": result.answer,\n                "calibrated": result.calibrated,\n                "prompt_version": result.prompt_version,\n', "val-cols")
out[p] = t

# ----------------------------------------------------------------- app.py
p = "app.py"
t = rd(p)
t = sub(t, '''from src.gate.reliability_gate import (
    EXPECTED_FEATURES,
    MODEL_NAME,
    N_SAMPLES,
    ReliabilityGate,
)

try:
    import torch
except Exception:  # torch is only used to report runtime status
    torch = None
''', '''from src.gate.config import EXPECTED_FEATURES, GATE_FILE_V4, MODEL_NAME, N_SAMPLES

# torch and ReliabilityGate are imported lazily (first "Check Reliability" click)
# so the first page render never waits on torch/transformers.
torch = None
''', "app-imports")
t = sub(t, "import sys\nfrom collections import Counter\n", "import sys\nimport threading\nfrom collections import Counter\n", "app-threading")
t = sub(t, '''@st.cache_resource(show_spinner=False)
def load_gate() -> ReliabilityGate:
    return ReliabilityGate()
''', '''@st.cache_resource(show_spinner=False)
def load_gate():
    # Imported here, not at module level: this pulls in torch + transformers.
    from src.gate.reliability_gate import ReliabilityGate

    return ReliabilityGate()
''', "app-loadgate")
t = sub(t, "def answer_groups(", '''@st.cache_resource(show_spinner=False)
def process_state() -> dict:
    """Status shared by every browser session of this server process."""
    return {
        "loaded": False,
        "error": None,
        "runtime": None,
        "threshold": None,
        # Cheap pre-load check: without a V4 gate file nothing can be calibrated.
        "calibrated": GATE_FILE_V4.exists(),
        "calibration_note": (
            "" if GATE_FILE_V4.exists() else
            "No gate trained with the current prompt (qa-useful-v4) exists yet, so any probability "
            "shown comes from the legacy gate and is not calibrated for this prompt. Every answer "
            "is shown as VERIFY until the recalibration workflow has been run and the app restarted. "
            "The response-quality safeguard still works."
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


def answer_groups(''', "app-state")
t = sub(t, "    decision = \"VERIFY\" if not quality.is_useful else model_decision\n",
        '''    calibrated = bool(getattr(result, "calibrated", True))
    decision = "VERIFY" if (not quality.is_useful or not calibrated) else model_decision
''', "app-decision")
t = sub(t, '        "quality_reason": quality.reason,\n',
        '        "quality_reason": quality.reason,\n        "calibrated": calibrated,\n', "app-viewdict")
t = sub(t, '        "loading": ("LOCAL MODEL • LOADING", "loading"),\n',
        '        "loading": ("LOCAL MODEL • LOADING", "loading"),\n        "idle": ("MODEL LOADS ON FIRST QUESTION", "loading"),\n', "app-hero-idle")
t = span(t, "def loading_html() -> str:", "def empty_state_html()", '''def loading_html(title: str, detail: str) -> str:
    """Honest, indeterminate loading state (no fake percentages or completed stages)."""
    return f"""
    <div class="loading card" role="status" aria-live="polite">
        <div class="spin" aria-hidden="true"></div>
        <div>
            <div class="ltitle">{esc(title)}</div>
            <div class="lsub">{esc(detail)}</div>
        </div>
        <div class="lbar" aria-hidden="true"></div>
    </div>
    """


''', "app-loading")
t = sub(t, '''        if not view["quality_ok"]:
            message = (
                "<b>Response-quality safeguard triggered.</b> "''', '''        if not view.get("calibrated", True):
            message = (
                "<b>Calibration refresh required.</b> This gate was not trained with the current "
                "prompt and generation settings, so its probability is not a calibrated estimate "
                "and every answer is shown as VERIFY."
                + ("" if view["quality_ok"] else f" Response-quality check: {esc(view['quality_reason'])}")
            )
        elif not view["quality_ok"]:
            message = (
                "<b>Response-quality safeguard triggered.</b> "''', "app-msg")
t = sub(t, '''    if view["quality_ok"]:
        comparison_caption = (
            f"{probability * 100:.1f}% {comparator} {thr_text}% → "
            f'<em class="{state}">{decision}</em>'
        )
    else:
        comparison_caption = (
            'Response-quality check → <em class="verify">VERIFY</em>'
        )
''', '''    calibrated = view.get("calibrated", True)
    chip_text = f"Trust threshold: {thr_text}%" if calibrated else "Calibration refresh required"
    if not calibrated:
        comparison_caption = (
            'Uncalibrated legacy estimate → <em class="verify">VERIFY</em>'
        )
    elif view["quality_ok"]:
        comparison_caption = (
            f"{probability * 100:.1f}% {comparator} {thr_text}% → "
            f'<em class="{state}">{decision}</em>'
        )
    else:
        comparison_caption = (
            'Response-quality check → <em class="verify">VERIFY</em>'
        )
''', "app-caption")
t = sub(t, '<div class="thr-chip">Trust threshold: {thr_text}%</div>', '<div class="thr-chip">{chip_text}</div>', "app-chip")
t = sub(t, "def sidebar_html(threshold: float, history: list[dict]) -> str:\n    thr_text = pct_plain(threshold)\n", '''def sidebar_html(threshold: float | None, history: list[dict]) -> str:
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
''', "app-side1")
t = sub(t, '<div class="side-kv"><span>Threshold</span><b>{thr_text}%</b></div>',
        '<div class="side-kv"><span>Threshold</span><b>{thr_display}</b></div>', "app-side2")
t = span(t, '    <div class="policy">\n        <small>Decision policy</small>', "    <div class=\"side-note\">",
         "    {policy_html}\n", "app-side3")
t = span(t, 'hero_slot = st.empty()\nhero_slot.markdown(html_block(hero_html("loading")), unsafe_allow_html=True)',
         "# ---------------- input card", '''# The page renders immediately. Qwen is NOT loaded here; see the click handler below.
_state = process_state()
load_error = _state["error"]

hero_slot = st.empty()
_hero_status = "online" if _state["loaded"] else ("offline" if _state["error"] else "idle")
hero_slot.markdown(
    html_block(hero_html(_hero_status, _state["runtime"] if _state["loaded"] else None)),
    unsafe_allow_html=True,
)

if not _state["calibrated"]:
    render(notice_html("warn", "Calibration refresh required", _state["calibration_note"]))


''', "app-hero")
t = span(t, "notice_slot = st.empty()\nstatus_slot = st.empty()", "# ---------------- model-load error detail", '''notice_slot = st.empty()
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

    elif not state["lock"].acquire(blocking=False):
        # Another click/session is mid-inference; never run two generations on the GPU at once.
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
            if not state["loaded"]:
                hero_slot.markdown(html_block(hero_html("loading")), unsafe_allow_html=True)
                status_slot.markdown(
                    html_block(
                        loading_html(
                            f"Loading {MODEL_NAME.split('/')[-1]} into memory…",
                            "First question only. Later questions reuse the loaded model.",
                        )
                    ),
                    unsafe_allow_html=True,
                )

            gate, load_error = ensure_gate()

            if gate is None:
                hero_slot.markdown(html_block(hero_html("offline")), unsafe_allow_html=True)
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
                status_slot.markdown(
                    html_block(
                        loading_html(
                            "Running the reliability gate…",
                            f"Generating {N_SAMPLES} sampled answers locally, then scoring them. "
                            "No cloud API involved.",
                        )
                    ),
                    unsafe_allow_html=True,
                )

                try:
                    new_view = run_check(gate, question_text)

                except (AttributeError, TypeError, ValueError, KeyError):
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
            state["lock"].release()


''', "app-run")
t = sub(t, "            gate.threshold if gate is not None else DEFAULT_THRESHOLD,\n",
        '            process_state()["threshold"] if process_state()["calibrated"] else None,\n', "app-sidebar-call")
out[p] = t

# ----------------------------------------------------------------- write everything
for rel, text in out.items():
    f = ROOT / rel
    bak = f.with_name(f.name + ".bak_v3")
    if not bak.exists():
        shutil.copy2(f, bak)
    f.write_text(text, encoding="utf-8")
    print("patched", rel)

for rel in ("src/gate/config.py", "scripts/benchmark_300_v4.py", "scripts/finalize_gate_v4.py"):
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "files" / rel, dst)
    print("added  ", rel)

print("\nAll edits applied. Next: python -m py_compile on the changed files (see workflow).")
