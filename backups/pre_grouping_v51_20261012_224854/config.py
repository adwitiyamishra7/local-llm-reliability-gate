"""Single source of truth for generation settings, prompt version and paths.

Deliberately has NO torch/transformers imports so the Streamlit UI can import it
instantly. Benchmark, live inference, finalization and the app all read from here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
CLASSIFIER_DIR = ROOT_DIR / "artifacts" / "classifier"
GATE_FILE_V4 = CLASSIFIER_DIR / "reliability_gate_final_v4.pkl"  # preserved, never overwritten
GATE_FILE_V5 = CLASSIFIER_DIR / "reliability_gate_final_v5.pkl"  # must be produced by a V5 benchmark + finalizer
GATE_FILE_LEGACY = CLASSIFIER_DIR / "reliability_gate_final.pkl"  # historical V3 artifact

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPT_VERSION = "qa-useful-v5"
SYSTEM_PROMPT = (
    "You are a concise factual question-answering assistant. "
    "Answer the exact question with the information requested; "
    "do not merely repeat a name or the words in the question. "
    "For a question like 'Who is [person]?', identify the person "
    "in one short sentence, including their role or what they are "
    "known for, rather than returning the name alone. "
    "For a question like 'What is [thing]?', briefly define or "
    "identify the thing. For calculations, clearly give the result. "
    "For questions asking only for a name, date, number, or place, "
    "give that value directly. Usually use one concise sentence. "
    "Do not use markdown or show step-by-step reasoning. "
    "Answer well-known people, places and facts directly from your knowledge. "
    "Do not guess names, dates or numbers you do not know; only if you do not "
    "recognise the subject at all, say briefly that you are not familiar with it."
)
# V5: the V4 prompt ended with a quoted refusal phrase ('I don't know'); a 0.6B model
# tends to copy a quoted literal. V5 permits honest abstention without quoting a phrase.

N_SAMPLES = 5
# 24 tokens truncated ordinary one-sentence answers (photosynthesis, Python, Ratan Tata).
# 48 covers a normal sentence; sampling stops early at EOS so short answers stay cheap.
MAX_NEW_TOKENS = 48
DO_SAMPLE = True
# Qwen3 non-thinking-mode recommended sampling (0.7 / 0.8 / 20); V4 used 0.8 / 0.90 / 50.
TEMPERATURE = 0.7
TOP_P = 0.80
TOP_K = 20

EXPECTED_FEATURES = [
    "mean_logprob_avg",
    "mean_logprob_std",
    "mean_logprob_best",
    "mean_logprob_worst",
    "min_logprob_avg",
    "min_logprob_std",
    "min_logprob_best",
    "min_logprob_worst",
    "logprob_confidence_range",
    "answer_agreement",
    "consensus_margin",
    "answer_key_entropy",
    "answer_length_avg",
    "answer_length_std",
    "truncation_rate",
]


def generation_config() -> dict:
    """Everything that defines the feature distribution the classifier sees."""
    return {
        "model_name": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "n_samples": N_SAMPLES,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "features": list(EXPECTED_FEATURES),
        "logprob_source": "raw_logits_log_softmax",
        # v2: truncation_rate = bad word ending OR generation ended without an EOS token
        "truncation_rule": "bad_ending_or_no_eos_v2",
        # V5: decoded text has <think> blocks stripped; consensus prefers USEFUL answers.
        # No retry/resampling: every one of the N_SAMPLES answers is used exactly as generated.
        "answer_cleaning": "strip_think_v1",
        "quality_rule": "response_quality_v2",
        # V5 repair: equivalent answer text is grouped conservatively, but distinct
        # useful candidates remain unresolved instead of being trusted by plurality.
        "answer_grouping_rule": "normalized_containment_and_question_subject_v1",
        "consensus_rule": "useful_plurality_with_disagreement_v2",
        "retry_policy": "none",
    }


def config_fingerprint(cfg: dict | None = None) -> str:
    cfg = generation_config() if cfg is None else cfg
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def config_differences(saved: dict, current: dict) -> list[str]:
    keys = sorted(set(saved) | set(current))
    return [
        f"{key}: trained={saved.get(key)!r} live={current.get(key)!r}"
        for key in keys
        if key != "features" and saved.get(key) != current.get(key)
    ] + (["features differ"] if saved.get("features") != current.get("features") else [])
