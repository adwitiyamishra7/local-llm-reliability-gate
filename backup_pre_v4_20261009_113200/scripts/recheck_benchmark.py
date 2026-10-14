from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "results"
    / "benchmark_results_50.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "artifacts"
    / "results"
    / "benchmark_results_50_clean.csv"
)


from src.gate.agreement import (
    extract_answer_key,
    normalize_answer,
)


# ============================================================
# EXPECTED ANSWER NORMALIZATION
# ============================================================

def expected_aliases(
    expected: str,
) -> list[str]:

    aliases = []

    for item in expected.split("|"):

        value = normalize_answer(item)

        if value:
            aliases.append(value)

    return aliases


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_signed_numbers(
    text: str,
) -> set[str]:

    values = re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?",
        text,
    )

    return {
        value.replace(",", "")
        for value in values
    }


# ============================================================
# EXPECTED MATCHING
# ============================================================

def answer_matches_expected(
    answer: str,
    expected: str,
) -> bool:

    answer_normalized = normalize_answer(
        answer
    )

    aliases = expected_aliases(
        expected
    )

    if not answer_normalized:
        return False

    # --------------------------------------------------------
    # Exact string match
    # --------------------------------------------------------

    if answer_normalized in aliases:
        return True

    # --------------------------------------------------------
    # Text aliases
    # --------------------------------------------------------

    for alias in aliases:

        # Do not use substring matching for pure numbers.
        if re.fullmatch(
            r"-?\d+(?:\.\d+)?",
            alias,
        ):
            continue

        # Reasonably long textual answer
        if len(alias) >= 3:

            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                answer_normalized,
            ):
                return True

    # --------------------------------------------------------
    # Numeric matching
    # --------------------------------------------------------

    expected_numbers = (
        extract_signed_numbers(expected)
    )

    answer_numbers = (
        extract_signed_numbers(answer)
    )

    if expected_numbers:

        if expected_numbers.intersection(
            answer_numbers
        ):
            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FINAL RECHECK OF 50-QUESTION BENCHMARK")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark results not found:\n"
            f"{INPUT_FILE}"
        )

    rows = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            question = (
                row.get("question")
                or ""
            ).strip()

            expected = (
                row.get("expected_answer")
                or ""
            ).strip()

            raw_answers = []

            for i in range(1, 6):

                answer = (
                    row.get(
                        f"raw_answer_{i}"
                    )
                    or ""
                ).strip()

                raw_answers.append(answer)

            # =================================================
            # CLEAN ANSWER KEYS
            # =================================================

            clean_keys = [
                extract_answer_key(
                    question,
                    answer,
                )
                for answer in raw_answers
            ]

            key_counts = Counter(
                clean_keys
            )

            consensus_key = (
                key_counts.most_common(1)[0][0]
            )

            consensus_answer = ""

            for answer, key in zip(
                raw_answers,
                clean_keys,
            ):

                if key == consensus_key:

                    consensus_answer = answer
                    break

            # =================================================
            # AGREEMENT
            # =================================================

            agreement = (
                key_counts.most_common(1)[0][1]
                / len(clean_keys)
            )

            # =================================================
            # CORRECTNESS
            # =================================================

            consensus_correct = (
                answer_matches_expected(
                    consensus_answer,
                    expected,
                )
            )

            # =================================================
            # DIAGNOSTIC ONLY
            #
            # This is NOT used as a classifier feature.
            # =================================================

            sample_correctness = [
                int(
                    answer_matches_expected(
                        answer,
                        expected,
                    )
                )
                for answer in raw_answers
            ]

            correct_sample_count = sum(
                sample_correctness
            )

            # =================================================
            # WRITE CLEAN FIELDS
            # =================================================

            for i in range(5):

                row[
                    f"clean_answer_key_{i + 1}"
                ] = clean_keys[i]

            row[
                "clean_answer_agreement"
            ] = agreement

            row[
                "clean_consensus_key"
            ] = consensus_key

            row[
                "clean_consensus_answer"
            ] = consensus_answer

            row[
                "clean_consensus_correct"
            ] = int(consensus_correct)

            row[
                "correct_sample_count"
            ] = correct_sample_count

            rows.append(row)

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        rows[0].keys()
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(rows)

    correct_count = sum(
        int(
            row[
                "clean_consensus_correct"
            ]
        )
        for row in rows
    )

    agreements = [
        float(
            row[
                "clean_answer_agreement"
            ]
        )
        for row in rows
    ]

    mean_agreement = (
        sum(agreements)
        / len(agreements)
    )

    print()
    print(
        f"Questions rechecked: {total}"
    )

    print(
        f"Correct consensus: "
        f"{correct_count}/{total}"
    )

    print(
        f"Benchmark accuracy: "
        f"{correct_count / total:.2%}"
    )

    print(
        f"Mean answer agreement: "
        f"{mean_agreement:.2f}"
    )

    print()
    print(
        f"Saved clean results:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # WRONG CASES
    # ========================================================

    print()
    print("=" * 65)
    print("REMAINING WRONG CONSENSUS CASES")
    print("=" * 65)

    wrong_count = 0

    for row in rows:

        if int(
            row[
                "clean_consensus_correct"
            ]
        ) == 0:

            wrong_count += 1

            print()
            print(
                f"{row['question_id']}: "
                f"{row['question']}"
            )

            print(
                f"Expected: "
                f"{row['expected_answer']}"
            )

            print(
                f"Consensus: "
                f"{row['clean_consensus_answer']}"
            )

            print(
                f"Key: "
                f"{row['clean_consensus_key']}"
            )

            print(
                f"Agreement: "
                f"{float(row['clean_answer_agreement']):.2f}"
            )

            print(
                f"Correct samples: "
                f"{row['correct_sample_count']}/5"
            )

    print()

    print(
        f"Wrong consensus cases: "
        f"{wrong_count}"
    )

    print()
    print("=" * 65)
    print("FINAL RECHECK FINISHED")
    print("=" * 65)


if __name__ == "__main__":
    main()