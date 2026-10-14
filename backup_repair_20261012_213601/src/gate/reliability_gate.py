from __future__ import annotations

"""
Live reliability gate for the Local LLM Reliability Gate project.

Pipeline:
question
    -> 5 sampled Qwen3-0.6B answers
    -> answer-key normalization
    -> 15 uncertainty features
    -> calibrated ExtraTrees model
    -> P(correct)
    -> TRUST / VERIFY

The feature calculations intentionally mirror benchmark_300.py so that
the live model receives the same feature representation used during training.
"""

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import pandas as pd

# Project root: .../local-llm-reliability-gate
ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.model import load_local_llm
from src.gate.generation import GenerationResult, generate_one
from src.gate.agreement import extract_answer_key
from src.gate.response_quality import assess_response_quality


from src.gate.config import (
    EXPECTED_FEATURES,
    GATE_FILE_LEGACY,
    GATE_FILE_V4,
    GATE_FILE_V5,
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

FINAL_GATE_FILE = GATE_FILE_V5


@dataclass
class GateResult:
    question: str
    answer: str
    probability_correct: float
    decision: str
    threshold: float

    answer_agreement: float
    consensus_margin: float
    answer_key_entropy: float

    mean_logprob_avg: float
    min_logprob_avg: float

    raw_answers: list[str]

    calibrated: bool = True
    calibration_note: str = ""
    prompt_version: str = PROMPT_VERSION

    # V5 response-quality fields (appended so older positional use is unaffected)
    quality_ok: bool = True
    quality_reason: str = ""
    reliability_reported: bool = True  # False -> probability_correct is diagnostic only
    useful_samples: int = 0
    nonanswer_majority: bool = False
    sample_useful: list[bool] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _shannon_entropy(keys: list[str]) -> float:
    counts = Counter(keys)
    n = len(keys)

    if n == 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return entropy


def _truncation_flag(answer: str) -> int:
    answer = answer.strip()

    if not answer:
        return 1

    bad_endings = (
        "$",
        "\\",
        "×",
        "is",
        "are",
        "was",
        "were",
        "and",
        "or",
        "of",
        "to",
        "the",
        ":",
    )

    if answer.endswith(bad_endings):
        return 1

    if answer.count("$") % 2 == 1:
        return 1

    return 0


def sample_answers(llm, question: str) -> list[GenerationResult]:
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


class ReliabilityGate:
    """Load the final calibrated gate and run live reliability inference."""

    def __init__(
        self,
        gate_file: Path | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        if gate_file is None:
            # V5 first. A V4/legacy file is still loadable but is flagged NOT calibrated below
            # (its recorded prompt/generation fingerprint no longer matches).
            if GATE_FILE_V5.exists():
                gate_file = GATE_FILE_V5
            elif GATE_FILE_V4.exists():
                gate_file = GATE_FILE_V4
            else:
                gate_file = GATE_FILE_LEGACY
        self.gate_file = Path(gate_file)

        if not self.gate_file.exists():
            raise FileNotFoundError(
                "Final reliability model not found:\n"
                f"{self.gate_file}\n\n"
                "Run the V5 benchmark and finalizer first."
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
                "Feature mismatch between training and live gate.\n"
                f"Expected: {EXPECTED_FEATURES}\n"
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
                f"current prompt ({PROMPT_VERSION}). Re-run the V5 recalibration workflow."
            )
        diffs = config_differences(saved, generation_config())
        if diffs:
            return False, (
                "Live prompt/generation settings differ from the ones this gate was trained "
                "with: " + "; ".join(diffs)
            )
        return True, ""

    @staticmethod
    def _build_features(
        question: str,
        results: list[GenerationResult],
    ) -> dict[str, float]:
        if not results:
            raise ValueError("At least one generation result is required.")

        answers = [result.answer.strip() for result in results]
        keys = [
            extract_answer_key(question, answer)
            for answer in answers
        ]

        counts = Counter(keys)

        winner_key, winner_count = counts.most_common(1)[0]
        second_count = (
            counts.most_common(2)[1][1]
            if len(counts) > 1
            else 0
        )

        agreement = winner_count / len(results)
        margin = (winner_count - second_count) / len(results)
        entropy = _shannon_entropy(keys)

        mean_lps = [
            float(result.mean_logprob)
            for result in results
            if result.mean_logprob is not None
        ]

        min_lps = [
            float(result.min_logprob)
            for result in results
            if result.min_logprob is not None
        ]

        if not mean_lps or not min_lps:
            raise ValueError(
                "No valid token log-probability values were produced."
            )

        mean_avg = sum(mean_lps) / len(mean_lps)
        mean_std = (
            sum((value - mean_avg) ** 2 for value in mean_lps)
            / len(mean_lps)
        ) ** 0.5

        min_avg = sum(min_lps) / len(min_lps)
        min_std = (
            sum((value - min_avg) ** 2 for value in min_lps)
            / len(min_lps)
        ) ** 0.5

        lengths = [len(answer.split()) for answer in answers]
        length_avg = sum(lengths) / len(lengths)

        length_std = (
            sum((value - length_avg) ** 2 for value in lengths)
            / len(lengths)
        ) ** 0.5

        # Truncated = suspicious word ending OR generation stopped without an end token.
        truncation_rate = (
            sum(
                1 if (_truncation_flag(answer) or not getattr(result, "finished", True)) else 0
                for answer, result in zip(answers, results)
            )
            / len(answers)
        )

        features = {
            "mean_logprob_avg": float(mean_avg),
            "mean_logprob_std": float(mean_std),
            "mean_logprob_best": float(max(mean_lps)),
            "mean_logprob_worst": float(min(mean_lps)),
            "min_logprob_avg": float(min_avg),
            "min_logprob_std": float(min_std),
            "min_logprob_best": float(max(min_lps)),
            "min_logprob_worst": float(min(min_lps)),
            "logprob_confidence_range": float(
                max(mean_lps) - min(mean_lps)
            ),
            "answer_agreement": float(agreement),
            "consensus_margin": float(margin),
            "answer_key_entropy": float(entropy),
            "answer_length_avg": float(length_avg),
            "answer_length_std": float(length_std),
            "truncation_rate": float(truncation_rate),
        }

        # Keeps the local calculation explicit and avoids a silent mismatch
        # between benchmark and live inference.
        if winner_key is None:
            raise RuntimeError("Consensus answer key could not be determined.")

        return features

    @staticmethod
    def _select_consensus(
        answers: list[str],
        question: str,
    ) -> tuple[str, dict]:
        """Pick the displayed answer from the largest group of USEFUL answers.

        Refusals/non-answers can never win just by being repeated. Ties between useful groups
        go to the group seen first (Counter keeps insertion order), not to the longest or most
        confident text. If NO sample is useful, the most common raw response is returned
        for display with quality_ok=False. Original answers are never modified.
        """
        if not answers:
            return "", {"quality_ok": False, "quality_reason": "No answers were generated.",
                        "useful_samples": 0, "nonanswer_majority": True, "flags": []}

        keys = [extract_answer_key(question, a) for a in answers]
        qualities = [assess_response_quality(question, a) for a in answers]
        flags = [q.is_useful for q in qualities]
        useful_counts = Counter(k for k, ok in zip(keys, flags) if ok)

        if not useful_counts:
            winner = Counter(keys).most_common(1)[0][0]
            idx = keys.index(winner)
            return answers[idx], {"quality_ok": False, "quality_reason": qualities[idx].reason,
                                  "useful_samples": 0, "nonanswer_majority": True, "flags": flags}

        winner, win_count = useful_counts.most_common(1)[0]
        idx = next(i for i, (k, ok) in enumerate(zip(keys, flags)) if ok and k == winner)
        non_useful = len(answers) - sum(flags)
        return answers[idx], {
            "quality_ok": True,
            "quality_reason": qualities[idx].reason,
            "useful_samples": sum(flags),
            "nonanswer_majority": non_useful > win_count,
            "flags": flags,
        }

    @staticmethod
    def _consensus_answer(
        answers: list[str],
        question: str,
    ) -> str:
        return ReliabilityGate._select_consensus(answers, question)[0]

    def predict(
        self,
        question: str,
        return_all_answers: bool = True,
    ) -> GateResult:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty.")

        results = sample_answers(self.llm, question)

        features = self._build_features(
            question=question,
            results=results,
        )

        feature_frame = pd.DataFrame(
            [[features[name] for name in self.feature_names]],
            columns=self.feature_names,
        )

        probability = float(
            self.model.predict_proba(feature_frame)[0, 1]
        )

        answers = [result.answer.strip() for result in results]

        answer, info = self._select_consensus(answers, question)

        # TRUST requires ALL of: valid calibration, a useful consensus answer, no non-answer
        # majority among the samples, and probability >= threshold. The classifier score alone
        # can never override a quality failure or a stale calibration.
        decision = (
            "TRUST"
            if (
                self.calibrated
                and info["quality_ok"]
                and not info["nonanswer_majority"]
                and probability >= self.threshold
            )
            else "VERIFY"
        )

        if not return_all_answers:
            answers = []

        return GateResult(
            question=question,
            answer=answer,
            probability_correct=probability,
            decision=decision,
            threshold=self.threshold,
            answer_agreement=features["answer_agreement"],
            consensus_margin=features["consensus_margin"],
            answer_key_entropy=features["answer_key_entropy"],
            mean_logprob_avg=features["mean_logprob_avg"],
            min_logprob_avg=features["min_logprob_avg"],
            raw_answers=answers,
            calibrated=self.calibrated,
            calibration_note=self.calibration_note,
            prompt_version=PROMPT_VERSION,
            quality_ok=info["quality_ok"],
            quality_reason=info["quality_reason"],
            reliability_reported=bool(self.calibrated and info["quality_ok"]),
            useful_samples=info["useful_samples"],
            nonanswer_majority=info["nonanswer_majority"],
            sample_useful=info["flags"],
        )
