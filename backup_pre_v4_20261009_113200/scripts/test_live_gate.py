from __future__ import annotations

"""
Smoke test for the live Local LLM Reliability Gate.

This test performs real local Qwen inference:
5 sampled answers for one question -> gate prediction.

Run from project root:
    python -m py_compile scripts\test_live_gate.py
    python scripts\test_live_gate.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.reliability_gate import ReliabilityGate


def main() -> None:
    print("=" * 70)
    print("LOCAL LLM RELIABILITY GATE")
    print("LIVE GATE SMOKE TEST")
    print("=" * 70)

    gate = ReliabilityGate()

    question = "What is the capital of Australia?"

    print(f"\nQuestion: {question}")
    print(f"Generating {5} sampled answers...\n")

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
    print(f"P(correct)          : {result.probability_correct:.4f}")
    print(f"Trust threshold     : {result.threshold:.2f}")
    print(f"Decision            : {result.decision}")
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
