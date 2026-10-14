from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.model import load_local_llm
from src.gate.generation import generate_one
from src.gate.agreement import extract_answer_key, normalize_answer

MODEL_NAME = "Qwen/Qwen3-0.6B"
INPUT_FILE = ROOT_DIR / "data" / "sample" / "benchmark_questions_300.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "results"
OUTPUT_FILE = OUTPUT_DIR / "benchmark_results_300_v3.csv"

N_SAMPLES = 5
MAX_NEW_TOKENS = 24
TEMPERATURE = 0.8
TOP_P = 0.90
TOP_K = 50


def load_questions() -> list[dict]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Benchmark dataset not found:\n{INPUT_FILE}")
    out = []
    with INPUT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("id") or "").strip()
            q = (row.get("question") or "").strip()
            expected = (row.get("expected") or "").strip()
            category = (row.get("category") or "unknown").strip()
            difficulty = (row.get("difficulty") or "unknown").strip()
            if qid and q:
                out.append({"id": qid, "question": q, "expected": expected,
                            "category": category, "difficulty": difficulty})
    return out


def completed_ids() -> set[str]:
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with OUTPUT_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("question_id"):
                done.add(row["question_id"].strip())
    return done


def expected_aliases(expected: str) -> list[str]:
    return [normalize_answer(x) for x in expected.split("|") if normalize_answer(x)]


def numbers(text: str) -> set[str]:
    vals = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    return {v.replace(",", "") for v in vals}


def answer_matches_expected(answer: str, expected: str) -> bool:
    a = normalize_answer(answer)
    if not a:
        return False
    aliases = expected_aliases(expected)
    if a in aliases:
        return True

    # Text aliases: require word-boundary match, case/format already normalized.
    for alias in aliases:
        if re.fullmatch(r"-?\d+(?:\.\d+)?", alias):
            continue
        if len(alias) >= 3 and re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", a):
            return True

    # Numeric aliases: do not treat a number in the question text as a match.
    exp_nums = numbers(expected)
    ans_nums = numbers(answer)
    if exp_nums and exp_nums.intersection(ans_nums):
        return True
    return False


def shannon_entropy(keys: list[str]) -> float:
    counts = Counter(keys)
    n = len(keys)
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return h


def truncation_flag(answer: str) -> int:
    a = answer.strip()
    if not a:
        return 1
    bad_endings = ("$", "\\", "×", "is", "are", "was", "were", "and", "or", "of", "to", "the", ":")
    if a.endswith(bad_endings):
        return 1
    # Unbalanced LaTeX delimiters are a strong truncation indicator.
    if a.count("$") % 2 == 1:
        return 1
    return 0


FIELDNAMES = [
    "question_id", "question", "expected_answer", "category", "difficulty",
    *[f"raw_answer_{i}" for i in range(1, 6)],
    *[f"answer_key_{i}" for i in range(1, 6)],
    *[f"mean_logprob_{i}" for i in range(1, 6)],
    *[f"min_logprob_{i}" for i in range(1, 6)],
    *[f"sample_latency_{i}" for i in range(1, 6)],
    "mean_logprob_avg", "mean_logprob_std", "mean_logprob_best", "mean_logprob_worst",
    "min_logprob_avg", "min_logprob_std", "min_logprob_best", "min_logprob_worst",
    "logprob_confidence_range", "answer_agreement", "consensus_margin", "answer_key_entropy",
    "answer_length_avg", "answer_length_std", "truncation_rate",
    "consensus_key", "consensus_answer", "consensus_correct", "question_latency_seconds",
]


def main() -> None:
    print("=" * 65)
    print("LOCAL LLM RELIABILITY GATE")
    print("300-QUESTION BENCHMARK V3")
    print("=" * 65)

    questions = load_questions()
    done = completed_ids()
    remaining = [q for q in questions if q["id"] not in done]

    print(f"\nDataset questions: {len(questions)}")
    print(f"Previously completed: {len(done)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Samples/question: {N_SAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Sampling: temperature={TEMPERATURE}, top_p={TOP_P}, top_k={TOP_K}")

    if not remaining:
        print(f"\nBenchmark already complete.\nResults: {OUTPUT_FILE}")
        return

    llm = load_local_llm(model_name=MODEL_NAME, prefer_gpu=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0
    mode = "a" if exists else "w"
    with OUTPUT_FILE.open(mode, encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
            out_f.flush()

        for idx, row in enumerate(remaining, 1):
            qid = row["id"]
            question = row["question"]
            expected = row["expected"]

            print("\n" + "=" * 65)
            print(f"Question {idx}/{len(remaining)}")
            print(f"ID: {qid}")
            print(f"Question: {question}")
            print(f"Expected: {expected}")

            q_start = perf_counter()
            answers, keys = [], []
            mean_lps, min_lps, latencies = [], [], []

            for sample in range(N_SAMPLES):
                print(f"\nGenerating sample {sample + 1}/{N_SAMPLES}...")
                result = generate_one(
                    llm=llm,
                    question=question,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                )
                answer = (result.answer or "").strip()
                key = extract_answer_key(question, answer)
                mlp = float(result.mean_logprob)
                mip = float(result.min_logprob)
                lat = float(result.latency_seconds)
                answers.append(answer); keys.append(key)
                mean_lps.append(mlp); min_lps.append(mip); latencies.append(lat)
                print(f"Raw answer: {answer}")
                print(f"Answer key: {key}")
                print(f"Mean logprob: {mlp:.6f}")
                print(f"Min logprob: {mip:.6f}")
                print(f"Latency: {lat:.3f}s")

            counts = Counter(keys)
            consensus_key = counts.most_common(1)[0][0]
            winner_count = counts.most_common(1)[0][1]
            second_count = counts.most_common(2)[1][1] if len(counts) > 1 else 0
            consensus_margin = (winner_count - second_count) / N_SAMPLES
            agreement = winner_count / N_SAMPLES
            entropy = shannon_entropy(keys)

            consensus_answer = next((a for a, k in zip(answers, keys) if k == consensus_key), answers[0])
            consensus_correct = answer_matches_expected(consensus_answer, expected)

            mean_avg = float(sum(mean_lps) / len(mean_lps))
            mean_std = float((sum((x - mean_avg) ** 2 for x in mean_lps) / len(mean_lps)) ** 0.5)
            mean_best = max(mean_lps)
            mean_worst = min(mean_lps)
            min_avg = float(sum(min_lps) / len(min_lps))
            min_std = float((sum((x - min_avg) ** 2 for x in min_lps) / len(min_lps)) ** 0.5)
            min_best = max(min_lps)
            min_worst = min(min_lps)
            conf_range = mean_best - mean_worst
            lengths = [len(a.split()) for a in answers]
            len_avg = sum(lengths) / len(lengths)
            len_std = (sum((x - len_avg) ** 2 for x in lengths) / len(lengths)) ** 0.5
            trunc_rate = sum(truncation_flag(a) for a in answers) / N_SAMPLES
            q_latency = perf_counter() - q_start

            print("\n--- QUESTION SUMMARY ---")
            print(f"Consensus key: {consensus_key}")
            print(f"Answer agreement: {agreement:.2f}")
            print(f"Consensus margin: {consensus_margin:.2f}")
            print(f"Consensus answer: {consensus_answer}")
            print(f"Benchmark correctness: {'CORRECT' if consensus_correct else 'WRONG'}")
            print(f"Question latency: {q_latency:.3f}s")

            output = {
                "question_id": qid,
                "question": question,
                "expected_answer": expected,
                "category": row["category"],
                "difficulty": row["difficulty"],
                "mean_logprob_avg": mean_avg,
                "mean_logprob_std": mean_std,
                "mean_logprob_best": mean_best,
                "mean_logprob_worst": mean_worst,
                "min_logprob_avg": min_avg,
                "min_logprob_std": min_std,
                "min_logprob_best": min_best,
                "min_logprob_worst": min_worst,
                "logprob_confidence_range": conf_range,
                "answer_agreement": agreement,
                "consensus_margin": consensus_margin,
                "answer_key_entropy": entropy,
                "answer_length_avg": len_avg,
                "answer_length_std": len_std,
                "truncation_rate": trunc_rate,
                "consensus_key": consensus_key,
                "consensus_answer": consensus_answer,
                "consensus_correct": int(consensus_correct),
                "question_latency_seconds": q_latency,
            }
            for i in range(5):
                output[f"raw_answer_{i+1}"] = answers[i]
                output[f"answer_key_{i+1}"] = keys[i]
                output[f"mean_logprob_{i+1}"] = mean_lps[i]
                output[f"min_logprob_{i+1}"] = min_lps[i]
                output[f"sample_latency_{i+1}"] = latencies[i]

            writer.writerow(output)
            out_f.flush()
            print(f"Saved: {OUTPUT_FILE}")

    print("\n" + "=" * 65)
    print("300-QUESTION BENCHMARK V3 FINISHED")
    print("=" * 65)
    print(f"Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
