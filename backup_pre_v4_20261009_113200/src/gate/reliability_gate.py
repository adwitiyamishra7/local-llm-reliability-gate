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


MODEL_NAME = "Qwen/Qwen3-0.6B"

FINAL_GATE_FILE = (
    ROOT_DIR
    / "artifacts"
    / "classifier"
    / "reliability_gate_final.pkl"
)

N_SAMPLES = 5
MAX_NEW_TOKENS = 24
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


class ReliabilityGate:
    """Load the final calibrated gate and run live reliability inference."""

    def __init__(
        self,
        gate_file: Path = FINAL_GATE_FILE,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.gate_file = Path(gate_file)

        if not self.gate_file.exists():
            raise FileNotFoundError(
                "Final reliability model not found:\n"
                f"{self.gate_file}\n\n"
                "Run scripts\\finalize_gate_v3.py first."
            )

        payload = joblib.load(self.gate_file)

        if not isinstance(payload, dict):
            raise ValueError(
                "Invalid final gate file: expected a dictionary payload."
            )

        required_keys = {
            "model",
            "feature_names",
            "trust_threshold",
        }

        missing = required_keys - set(payload.keys())

        if missing:
            raise ValueError(
                "Final gate file is missing required fields: "
                + ", ".join(sorted(missing))
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

        # Load Qwen once. The model stays in memory for repeated questions.
        self.llm = load_local_llm(
            model_name=self.model_name,
            prefer_gpu=True,
        )

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

        truncation_rate = (
            sum(_truncation_flag(answer) for answer in answers)
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
    def _consensus_answer(
        answers: list[str],
        question: str,
    ) -> str:
        if not answers:
            return ""

        keys = [
            extract_answer_key(question, answer)
            for answer in answers
        ]

        winner = Counter(keys).most_common(1)[0][0]

        for answer, key in zip(answers, keys):
            if key == winner:
                return answer

        return answers[0]

    def predict(
        self,
        question: str,
        return_all_answers: bool = True,
    ) -> GateResult:
        question = question.strip()

        if not question:
            raise ValueError("Question must not be empty.")

        results: list[GenerationResult] = []

        for _ in range(N_SAMPLES):
            result = generate_one(
                llm=self.llm,
                question=question,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )

            results.append(result)

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

        decision = (
            "TRUST"
            if probability >= self.threshold
            else "VERIFY"
        )

        answers = [result.answer.strip() for result in results]

        answer = self._consensus_answer(
            answers=answers,
            question=question,
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
        )
