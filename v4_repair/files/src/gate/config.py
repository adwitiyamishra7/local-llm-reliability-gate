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
GATE_FILE_V4 = CLASSIFIER_DIR / "reliability_gate_final_v4.pkl"
GATE_FILE_LEGACY = CLASSIFIER_DIR / "reliability_gate_final.pkl"  # historical V3 artifact

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPT_VERSION = "qa-useful-v4"
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
    "If you genuinely do not know, say 'I don't know' rather than "
    "inventing an answer."
)

N_SAMPLES = 5
# 24 tokens truncated ordinary one-sentence answers (photosynthesis, Python, Ratan Tata).
# 48 covers a normal sentence; sampling stops early at EOS so short answers stay cheap.
MAX_NEW_TOKENS = 48
DO_SAMPLE = True
TEMPERATURE = 0.8
TOP_P = 0.90
TOP_K = 50

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
