from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.agreement import agreement_score, consensus_answer, normalize_answer
from src.gate.generation import generate_one
from src.gate.model import load_local_llm


INPUT_FILE = PROJECT_ROOT / "data" / "sample" / "pilot_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "artifacts" / "results" / "pilot_results.csv"

SAMPLES = 5


def normalize_expected_answers(raw_value: str) -> list[str]:
    """Convert pipe-separated expected answers into normalized answers."""
    return [
        normalize_answer(answer)
        for answer in raw_value.split("|")
        if answer.strip()
    ]


def save_result(row: dict) -> None:
    """
    Append one completed question to disk immediately.

    This makes the pilot resumable if the process is interrupted.
    """
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = OUTPUT_FILE.exists()

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys(),
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def load_completed_ids() -> set[str]:
    """Return question IDs already completed in a previous run."""
    if not OUTPUT_FILE.exists():
        return set()

    completed = set()

    with OUTPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            if row.get("question_id"):
                completed.add(row["question_id"])

    return completed


def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Pilot question file not found: {INPUT_FILE}"
        )

    print("=" * 65)
    print("LOCAL LLM RELIABILITY GATE")
    print("10-QUESTION PILOT EXPERIMENT")
    print("=" * 65)

    completed_ids = load_completed_ids()

    print(
        f"\nPreviously completed questions: "
        f"{len(completed_ids)}"
    )

    questions = []

    with INPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    remaining = [
        row
        for row in questions
        if row["id"] not in completed_ids
    ]

    print(
        f"Questions remaining: "
        f"{len(remaining)}"
    )

    if not remaining:
        print("\nPilot already completed.")
        print(f"Results: {OUTPUT_FILE}")
        return

    llm = load_local_llm()

    for question_index, row in enumerate(
        remaining,
        start=1,
    ):

        question_id = row["id"]
        question = row["question"]

        expected_answers = normalize_expected_answers(
            row["expected_answers"]
        )

        print("\n" + "=" * 65)
        print(
            f"Question {question_index}/{len(remaining)}"
        )
        print(f"ID: {question_id}")
        print(f"Question: {question}")

        results = []

        question_start = perf_counter()

        for sample_index in range(SAMPLES):

            print(
                f"\nGenerating sample "
                f"{sample_index + 1}/{SAMPLES}..."
            )

            result = generate_one(
                llm,
                question=question,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
            )

            results.append(result)

            print(
                f"Answer: {result.answer}"
            )

            print(
                "Mean logprob: "
                f"{result.mean_logprob:.6f}"
                if result.mean_logprob is not None
                else "Mean logprob: N/A"
            )

            print(
                f"Latency: "
                f"{result.latency_seconds:.3f}s"
            )

        question_latency = (
            perf_counter() - question_start
        )

        answers = [
            result.answer
            for result in results
        ]

        agreement = agreement_score(answers)
        consensus = consensus_answer(answers)

        normalized_consensus = normalize_answer(
            consensus
        )

        # This is deliberately a simple PILOT correctness check.
        # It is NOT our final evaluation method.
        consensus_correct = any(
            expected in normalized_consensus
            for expected in expected_answers
        )

        mean_logprobs = [
            result.mean_logprob
            for result in results
            if result.mean_logprob is not None
        ]

        min_logprobs = [
            result.min_logprob
            for result in results
            if result.min_logprob is not None
        ]

        average_mean_logprob = (
            sum(mean_logprobs) / len(mean_logprobs)
            if mean_logprobs
            else None
        )

        average_min_logprob = (
            sum(min_logprobs) / len(min_logprobs)
            if min_logprobs
            else None
        )

        output = {
            "question_id": question_id,
            "question": question,
            "expected_answers": "|".join(
                expected_answers
            ),

            "answer_1": answers[0],
            "answer_2": answers[1],
            "answer_3": answers[2],
            "answer_4": answers[3],
            "answer_5": answers[4],

            "mean_logprob_1": results[0].mean_logprob,
            "mean_logprob_2": results[1].mean_logprob,
            "mean_logprob_3": results[2].mean_logprob,
            "mean_logprob_4": results[3].mean_logprob,
            "mean_logprob_5": results[4].mean_logprob,

            "min_logprob_1": results[0].min_logprob,
            "min_logprob_2": results[1].min_logprob,
            "min_logprob_3": results[2].min_logprob,
            "min_logprob_4": results[3].min_logprob,
            "min_logprob_5": results[4].min_logprob,

            "average_mean_logprob": average_mean_logprob,
            "average_min_logprob": average_min_logprob,

            "consensus_answer": consensus,
            "agreement": agreement,

            "pilot_consensus_correct": int(
                consensus_correct
            ),

            "question_latency_seconds": question_latency,
        }

        save_result(output)

        print("\n--- QUESTION SUMMARY ---")
        print(f"Consensus: {consensus}")
        print(
            f"Agreement: "
            f"{agreement:.2f}"
        )
        print(
            "Pilot consensus correctness: "
            f"{'CORRECT' if consensus_correct else 'WRONG'}"
        )
        print(
            f"Question latency: "
            f"{question_latency:.3f}s"
        )
        print(
            f"Saved to: "
            f"{OUTPUT_FILE}"
        )

    print("\n" + "=" * 65)
    print("PILOT EXPERIMENT FINISHED")
    print("=" * 65)

    print(
        f"\nResults file:\n{OUTPUT_FILE}"
    )

    print(
        "\nIMPORTANT:"
        "\nThe automatic correctness field is only a pilot heuristic."
        "\nWe will manually inspect these 10 questions before using"
        "\nthis logic for the real benchmark."
    )


if __name__ == "__main__":
    main()