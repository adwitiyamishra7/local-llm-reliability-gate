from __future__ import annotations

import csv
import inspect
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.gate.model import load_local_llm
from src.gate.generation import generate_one
from src.gate.agreement import (
    agreement_score,
    extract_answer_key,
    normalize_answer,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "sample"
    / "benchmark_questions_50.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "artifacts"
    / "results"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "benchmark_results_50.csv"
)

N_SAMPLES = 5

MAX_NEW_TOKENS = 16

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

# ============================================================
# CSV HELPERS
# ============================================================

FIELDNAMES = [
    "question_id",
    "question",
    "expected_answer",

    "raw_answer_1",
    "raw_answer_2",
    "raw_answer_3",
    "raw_answer_4",
    "raw_answer_5",

    "answer_key_1",
    "answer_key_2",
    "answer_key_3",
    "answer_key_4",
    "answer_key_5",

    "mean_logprob_1",
    "mean_logprob_2",
    "mean_logprob_3",
    "mean_logprob_4",
    "mean_logprob_5",

    "min_logprob_1",
    "min_logprob_2",
    "min_logprob_3",
    "min_logprob_4",
    "min_logprob_5",

    "sample_latency_1",
    "sample_latency_2",
    "sample_latency_3",
    "sample_latency_4",
    "sample_latency_5",

    "mean_logprob_avg",
    "min_logprob_avg",

    "answer_agreement",
    "consensus_key",
    "consensus_answer",

    "pilot_consensus_correct",
    "question_latency_seconds",
]


def load_questions() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark questions not found:\n{INPUT_FILE}"
        )

    questions: list[dict] = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            question_id = (
                row.get("id")
                or row.get("question_id")
                or ""
            ).strip()

            question = (
                row.get("question")
                or ""
            ).strip()

            expected = (
                row.get("expected")
                or row.get("expected_answer")
                or row.get("answer")
                or ""
            ).strip()

            if not question_id or not question:
                continue

            questions.append(
                {
                    "id": question_id,
                    "question": question,
                    "expected": expected,
                }
            )

    return questions


def load_completed_ids() -> set[str]:
    completed: set[str] = set()

    if not OUTPUT_FILE.exists():
        return completed

    try:
        with OUTPUT_FILE.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                question_id = (
                    row.get("question_id")
                    or ""
                ).strip()

                if question_id:
                    completed.add(question_id)

    except Exception as exc:
        print(
            f"Warning: could not read existing benchmark results: {exc}"
        )

    return completed


def create_writer():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = OUTPUT_FILE.exists()

    file = OUTPUT_FILE.open(
        "a",
        encoding="utf-8",
        newline="",
    )

    writer = csv.DictWriter(
        file,
        fieldnames=FIELDNAMES,
    )

    if not file_exists or OUTPUT_FILE.stat().st_size == 0:
        writer.writeheader()
        file.flush()

    return file, writer


# ============================================================
# ANSWER KEY HELPER
# ============================================================

def get_answer_key(
    question: str,
    answer: str,
) -> str:
    """
    Calls extract_answer_key using the parameter order
    implemented in the current agreement.py.

    This small compatibility layer prevents this benchmark
    script from depending on argument ordering.
    """

    parameters = list(
        inspect.signature(
            extract_answer_key
        ).parameters.keys()
    )

    if len(parameters) < 2:
        return str(
            extract_answer_key(answer)
        ).strip()

    first_name = parameters[0].lower()
    second_name = parameters[1].lower()

    if "question" in first_name and "answer" in second_name:
        key = extract_answer_key(
            question,
            answer,
        )
    else:
        key = extract_answer_key(
            answer,
            question,
        )

    return str(key).strip()


# ============================================================
# EXPECTED ANSWER CHECK
# ============================================================

def expected_aliases(
    expected: str,
) -> set[str]:
    """
    Expected answers are stored as aliases separated by |.

    Example:
        "william shakespeare|shakespeare"
    """

    aliases = set()

    for item in expected.split("|"):
        cleaned = normalize_answer(item)

        if cleaned:
            aliases.add(cleaned)

    return aliases


def is_correct(
    consensus_key: str,
    expected: str,
) -> bool:
    expected_set = expected_aliases(expected)

    consensus_normalized = normalize_answer(
        consensus_key
    )

    return (
        consensus_normalized in expected_set
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 65)
    print("LOCAL LLM RELIABILITY GATE")
    print("50-QUESTION BENCHMARK")
    print("=" * 65)

    questions = load_questions()

    if not questions:
        raise ValueError(
            "No benchmark questions were loaded."
        )

    completed_ids = load_completed_ids()

    remaining = [
        row
        for row in questions
        if row["id"] not in completed_ids
    ]

    print()
    print(
        f"Previously completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    if not remaining:
        print()
        print("Benchmark already complete.")
        print(
            f"Results: {OUTPUT_FILE}"
        )
        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    llm = load_local_llm(
        model_name=MODEL_NAME,
        prefer_gpu=True,
    )

    # ========================================================
    # OPEN RESULT FILE
    # ========================================================

    result_file, writer = create_writer()

    try:

        for q_index, row in enumerate(
            remaining,
            start=1,
        ):

            question_id = row["id"]
            question = row["question"]
            expected = row["expected"]

            print()
            print("=" * 65)
            print(
                f"Question "
                f"{q_index}/{len(remaining)}"
            )
            print(
                f"ID: {question_id}"
            )
            print(
                f"Question: {question}"
            )
            print(
                f"Expected: {expected}"
            )

            question_start = perf_counter()

            raw_answers: list[str] = []
            answer_keys: list[str] = []

            mean_logprobs: list[float] = []
            min_logprobs: list[float] = []
            latencies: list[float] = []

            # =================================================
            # FIVE SAMPLED ANSWERS
            # =================================================

            for sample_index in range(
                N_SAMPLES
            ):

                print()
                print(
                    f"Generating sample "
                    f"{sample_index + 1}/{N_SAMPLES}..."
                )

                result = generate_one(
                    llm=llm,
                    question=question,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                )

                answer = (
                    result.answer
                    if result.answer
                    else ""
                ).strip()

                key = get_answer_key(
                    question,
                    answer,
                )

                raw_answers.append(answer)
                answer_keys.append(key)

                mean_lp = (
                    float(result.mean_logprob)
                    if result.mean_logprob
                    is not None
                    else float("nan")
                )

                min_lp = (
                    float(result.min_logprob)
                    if result.min_logprob
                    is not None
                    else float("nan")
                )

                latency = float(
                    result.latency_seconds
                )

                mean_logprobs.append(
                    mean_lp
                )

                min_logprobs.append(
                    min_lp
                )

                latencies.append(
                    latency
                )

                print(
                    f"Raw answer: {answer}"
                )

                print(
                    f"Answer key: {key}"
                )

                print(
                    f"Mean logprob: "
                    f"{mean_lp:.6f}"
                )

                print(
                    f"Min logprob: "
                    f"{min_lp:.6f}"
                )

                print(
                    f"Latency: "
                    f"{latency:.3f}s"
                )

            # =================================================
            # AGREEMENT
            # =================================================

            question_context = [
                question
            ] * len(raw_answers)

            answer_agreement = agreement_score(
                raw_answers,
                question_context,
            )

            # =================================================
            # CONSENSUS
            # =================================================

            key_counts = Counter(
                answer_keys
            )

            consensus_key = (
                key_counts.most_common(1)[0][0]
            )

            consensus_answer = ""

            for answer, key in zip(
                raw_answers,
                answer_keys,
            ):
                if key == consensus_key:
                    consensus_answer = answer
                    break

            # =================================================
            # CORRECTNESS
            # =================================================

            consensus_correct = is_correct(
                consensus_key,
                expected,
            )

            # =================================================
            # AGGREGATE CONFIDENCE
            # =================================================

            valid_mean = [
                value
                for value in mean_logprobs
                if value == value
            ]

            valid_min = [
                value
                for value in min_logprobs
                if value == value
            ]

            mean_logprob_avg = (
                sum(valid_mean)
                / len(valid_mean)
                if valid_mean
                else float("nan")
            )

            min_logprob_avg = (
                sum(valid_min)
                / len(valid_min)
                if valid_min
                else float("nan")
            )

            question_latency = (
                perf_counter()
                - question_start
            )

            # =================================================
            # PRINT SUMMARY
            # =================================================

            print()
            print(
                "--- QUESTION SUMMARY ---"
            )

            print(
                f"Consensus key: "
                f"{consensus_key}"
            )

            print(
                f"Answer agreement: "
                f"{answer_agreement:.2f}"
            )

            print(
                f"Consensus answer: "
                f"{consensus_answer}"
            )

            print(
                "Benchmark correctness: "
                f"{'CORRECT' if consensus_correct else 'WRONG'}"
            )

            print(
                f"Question latency: "
                f"{question_latency:.3f}s"
            )

            # =================================================
            # SAVE ROW
            # =================================================

            output_row = {
                "question_id": question_id,
                "question": question,
                "expected_answer": expected,

                "mean_logprob_avg": mean_logprob_avg,
                "min_logprob_avg": min_logprob_avg,

                "answer_agreement": answer_agreement,
                "consensus_key": consensus_key,
                "consensus_answer": consensus_answer,

                "pilot_consensus_correct": int(
                    consensus_correct
                ),

                "question_latency_seconds":
                    question_latency,
            }

            for i in range(N_SAMPLES):

                sample_number = i + 1

                output_row[
                    f"raw_answer_{sample_number}"
                ] = raw_answers[i]

                output_row[
                    f"answer_key_{sample_number}"
                ] = answer_keys[i]

                output_row[
                    f"mean_logprob_{sample_number}"
                ] = mean_logprobs[i]

                output_row[
                    f"min_logprob_{sample_number}"
                ] = min_logprobs[i]

                output_row[
                    f"sample_latency_{sample_number}"
                ] = latencies[i]

            writer.writerow(
                output_row
            )

            result_file.flush()

            print(
                f"Saved: {OUTPUT_FILE}"
            )

    finally:
        result_file.close()

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 65)
    print("50-QUESTION BENCHMARK FINISHED")
    print("=" * 65)

    print()
    print(
        f"Results: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()