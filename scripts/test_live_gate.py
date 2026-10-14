from __future__ import annotations

"""
Smoke test for the live Local LLM Reliability Gate.

This test performs real local Qwen inference:
5 sampled answers for one question -> gate prediction.

Run from project root:
    python -m py_compile scripts\test_live_gate.py
    python scripts\test_live_gate.py
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.reliability_gate import ReliabilityGate
from src.gate.config import N_SAMPLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real-model reliability-gate check.")
    parser.add_argument(
        "--question",
        default="What is the capital of Australia?",
        help="Question to run through the live five-sample gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("LOCAL LLM RELIABILITY GATE")
    print("LIVE GATE SMOKE TEST")
    print("=" * 70)

    gate = ReliabilityGate()

    question = args.question

    print(f"\nQuestion: {question}")
    print(f"Generating {N_SAMPLES} sampled answers...\n")

    result = gate.predict(question)

    print("-" * 70)
    print("SAMPLED ANSWERS")
    print("-" * 70)

    for index, answer in enumerate(result.raw_answers, start=1):
        print(f"{index}. {answer}")

    print("\n" + "-" * 70)
    print("GATE RESULT")
    print("-" * 70)
    print(f"Consensus answer    : {result.answer}")
    if result.reliability_reported:
        print(f"P(correct)          : {result.probability_correct:.4f}")
    else:
        print(
            "P(correct)          : N/A (reliability not reported; "
            f"raw diagnostic={result.probability_correct:.4f})"
        )
    print(f"Trust threshold     : {result.threshold:.2f}")
    print(f"Decision            : {result.decision}")
    print(f"Reliability reported: {result.reliability_reported}")
    print(f"Verifier status     : {result.fact_verification_status}")
    print(f"Verifier supported  : {result.fact_verification_supported}")
    print(f"Verifier reason     : {result.fact_verification_reason}")
    print(f"Material disagreement: {result.material_disagreement}")
    print(f"Calibrated          : {result.calibrated}")
    print(f"Answer agreement    : {result.answer_agreement:.2f}")
    print(f"Consensus margin    : {result.consensus_margin:.2f}")
    print(f"Answer-key entropy  : {result.answer_key_entropy:.4f}")
    print(f"Mean logprob avg    : {result.mean_logprob_avg:.6f}")
    print(f"Min logprob avg     : {result.min_logprob_avg:.6f}")

    print("\n" + "=" * 70)
    print("LIVE GATE SMOKE TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()
