from __future__ import annotations

"""Run a 10-question live validation set through Qwen + the frozen gate.

This performs 50 sampled generations (10 questions x 5 samples) and may take
several minutes on a GTX 1650. Results are saved to artifacts/results/.

Run from project root:
    python -m py_compile scripts\validate_live_gate.py
    python scripts\validate_live_gate.py
"""

import csv
import sys
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.reliability_gate import ReliabilityGate
from src.gate.response_quality import assess_response_quality

QUESTIONS = [
    "What is the capital of Australia?",
    "Who wrote Hamlet?",
    "What is 15% of 240?",
    "Who is Ratan Tata?",
    "Who discovered penicillin?",
    "What is the largest planet in the Solar System?",
    "What is the square root of 144?",
    "Who was the first president of the United States?",
    "Explain photosynthesis in one sentence.",
    "What is Python in computer science?",
]

OUTPUT = ROOT_DIR / "artifacts" / "results" / "live_validation_10.csv"


def main() -> None:
    print("=" * 76)
    print("LOCAL LLM RELIABILITY GATE - 10 QUESTION LIVE VALIDATION")
    print("=" * 76)
    print(f"Questions: {len(QUESTIONS)}")
    print("Generations expected: 50 (5 per question)")
    print("This performs real local Qwen inference.\n")

    gate = ReliabilityGate()
    rows: list[dict[str, object]] = []
    total_started = perf_counter()

    for index, question in enumerate(QUESTIONS, start=1):
        started = perf_counter()
        print("-" * 76)
        print(f"[{index:02d}/{len(QUESTIONS)}] {question}")

        try:
            result = gate.predict(question=question, return_all_answers=True)
            quality = assess_response_quality(question, result.answer)
            final_decision = "VERIFY" if not quality.is_useful else result.decision
            elapsed = perf_counter() - started

            row = {
                "index": index,
                "question": question,
                "consensus_answer": result.answer,
                "probability_correct": round(float(result.probability_correct), 6),
                "classifier_decision": result.decision,
                "response_quality": "PASS" if quality.is_useful else "FLAG",
                "quality_reason": quality.reason,
                "final_decision": final_decision,
                "answer_agreement": round(float(result.answer_agreement), 6),
                "consensus_margin": round(float(result.consensus_margin), 6),
                "answer_key_entropy": round(float(result.answer_key_entropy), 6),
                "latency_seconds": round(elapsed, 2),
                "sample_1": result.raw_answers[0] if len(result.raw_answers) > 0 else "",
                "sample_2": result.raw_answers[1] if len(result.raw_answers) > 1 else "",
                "sample_3": result.raw_answers[2] if len(result.raw_answers) > 2 else "",
                "sample_4": result.raw_answers[3] if len(result.raw_answers) > 3 else "",
                "sample_5": result.raw_answers[4] if len(result.raw_answers) > 4 else "",
            }
            rows.append(row)

            print(f"Consensus answer : {result.answer!r}")
            print(f"P(correct)       : {result.probability_correct:.4f}")
            print(f"Classifier       : {result.decision}")
            print(f"Quality check    : {'PASS' if quality.is_useful else 'FLAG'}")
            if not quality.is_useful:
                print(f"Quality reason   : {quality.reason}")
            print(f"Final decision   : {final_decision}")
            print(f"Agreement        : {result.answer_agreement:.2f}")
            print(f"Latency          : {elapsed:.1f}s")

        except Exception as exc:
            elapsed = perf_counter() - started
            print(f"ERROR: {type(exc).__name__}: {exc}")
            rows.append({
                "index": index,
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_seconds": round(elapsed, 2),
            })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_columns = sorted({key for row in rows for key in row.keys()})
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    flags = sum(row.get("response_quality") == "FLAG" for row in rows)
    errors = sum("error" in row for row in rows)
    print("\n" + "=" * 76)
    print("VALIDATION COMPLETE")
    print("=" * 76)
    print(f"Rows completed  : {len(rows) - errors}/{len(QUESTIONS)}")
    print(f"Quality flags   : {flags}")
    print(f"Errors          : {errors}")
    print(f"Total time      : {(perf_counter() - total_started) / 60:.1f} minutes")
    print(f"CSV saved       : {OUTPUT}")
    print("\nImportant: this validates response usefulness and gate behavior, not factual accuracy.")


if __name__ == "__main__":
    main()
