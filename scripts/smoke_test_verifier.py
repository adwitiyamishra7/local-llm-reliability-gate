"""Cheap factual-verifier smoke test. Does not load Qwen or a classifier.

Run from project root:
    python scripts/smoke_test_verifier.py

The arithmetic checks are fully local. The capital check needs Internet access to Wikidata.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gate.factual_verifier import verify_answer  # noqa: E402


def main() -> int:
    checks = [
        ("What is 15% of 240?", "15% of 240 is 36.", "VERIFIED"),
        ("What is 15% of 240?", "15% of 240 is 35.", "CONTRADICTED"),
        ("Who is Isaac Newton?", "Isaac Newton was an English physicist.", "UNSUPPORTED"),
    ]
    failed = 0
    for question, answer, expected in checks:
        result = verify_answer(question, answer)
        ok = result.status == expected
        print(
            f"{'PASS' if ok else 'FAIL'} local: expected={expected}, actual={result.status} "
            f"| reason={result.reason} | {question}"
        )
        if not ok:
            failed += 1

    question = "What is the capital of Australia?"
    result = verify_answer(question, "The capital of Australia is Canberra.")
    if result.status == "UNAVAILABLE":
        print(
            "BLOCKED live: expected=VERIFIED, actual=UNAVAILABLE "
            f"| reason={result.reason}"
        )
        failed += 1
    else:
        ok = result.status == "VERIFIED"
        print(
            f"{'PASS' if ok else 'FAIL'} live: expected=VERIFIED, actual={result.status} "
            f"| reason={result.reason}"
        )
        if result.source_url:
            print(f"  source: {result.source_url}")
        if result.evidence:
            print(f"  evidence: {result.evidence}")
        if not ok:
            failed += 1

    print(f"\nVerifier smoke result: {4 - failed}/4 checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
