from __future__ import annotations

"""Regenerate the 300-question benchmark with the CURRENT live configuration.

* Uses the same prompt, sampling settings, sample count, max length, log-prob
  calculation and feature code as live inference (sample_answers +
  ReliabilityGate._build_features).
* Reads questions + reference answers from the existing V3 benchmark file
  (no invented questions/labels). Never writes to the V3 file.
* Resumable: rows are appended as they finish; re-run the same command to continue.

    python scripts\\benchmark_300_v4.py --limit 5 --output benchmark_smoke_v4.csv   (smoke test)
    python scripts\\benchmark_300_v4.py
"""

import argparse
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.config import MODEL_NAME, PROMPT_VERSION, config_fingerprint
from src.gate.model import load_local_llm
from src.gate.reliability_gate import ReliabilityGate, sample_answers

RESULTS_DIR = ROOT_DIR / "artifacts" / "results"
V3_FILE = RESULTS_DIR / "benchmark_results_300_v3.csv"
TARGET = "consensus_correct"
QUESTION_COLS = ("question", "prompt", "query")
REFERENCE_COLS = (
    "reference_answer", "reference", "ground_truth",
    "gold_answer", "expected_answer", "correct_answer",
)
V3_ANSWER_COLS = ("consensus_answer", "consensus", "model_answer")


def _norm(text: object) -> str:
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    t = str(text).translate(subscripts).lower().replace("−", "-").replace("﹣", "-")
    # Normalize common math/LaTeX rendering without discarding semantic signs.
    t = re.sub(r"\\(log|text|mathrm)\s*", r"\1 ", t)
    t = t.replace("\\", " ").replace(",", "")
    t = re.sub(r"[^a-z0-9.%+\-\s]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)  # drop sentence periods, keep decimals
    return " ".join(t.split())


def label_correct(answer: object, reference: object) -> int:
    """Match a curated reference alias, preserving numeric signs and decimal values."""
    ans = _norm(answer)
    number_pattern = re.compile(r"(?<![a-z0-9.+\-])[+-]?\d+(?:\.\d+)?%?(?![a-z0-9.])")
    answer_numbers: set[Decimal] = set()
    for token in number_pattern.findall(ans):
        try:
            answer_numbers.add(Decimal(token.rstrip("%")))
        except InvalidOperation:
            pass
    for ref in re.split(r"\s*(?:\||;)\s*", str(reference)):
        r = _norm(ref)
        if not r:
            # Punctuation-only references (notably Python's '#') remain literal.
            if str(ref).strip() and str(ref).strip() in str(answer):
                return 1
            continue
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", r):
            try:
                if Decimal(r.rstrip("%")) in answer_numbers:
                    return 1
            except InvalidOperation:
                pass
        elif re.search(rf"(?<![a-z0-9]){re.escape(r)}(?![a-z0-9])", ans):
            return 1
    return 0


def _pick(df: pd.DataFrame, candidates: tuple[str, ...], override: str | None, what: str) -> str:
    if override:
        if override not in df.columns:
            raise SystemExit(f"--{what}-col {override!r} not in columns: {list(df.columns)}")
        return override
    for name in candidates:
        if name in df.columns:
            return name
    raise SystemExit(
        f"Could not find the {what} column. Columns are:\n  {list(df.columns)}\n"
        f"Pass --{what}-col <name>."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(V3_FILE))
    ap.add_argument("--output", default="benchmark_results_300_v4.csv")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--question-col", default=None)
    ap.add_argument("--reference-col", default=None)
    ap.add_argument("--skip-label-check", action="store_true")
    args = ap.parse_args()

    output = RESULTS_DIR / args.output
    if output.resolve() == V3_FILE.resolve():
        raise SystemExit("Refusing to overwrite the V3 benchmark.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    src = pd.read_csv(args.dataset)
    qcol = _pick(src, QUESTION_COLS, args.question_col, "question")
    rcol = _pick(src, REFERENCE_COLS, args.reference_col, "reference")
    print(f"Dataset: {args.dataset}  rows={len(src)}  question={qcol!r}  reference={rcol!r}")

    # Self-check: does this labeling rule reproduce the V3 labels on V3's own answers?
    acol = next((c for c in V3_ANSWER_COLS if c in src.columns), None)
    if acol and TARGET in src.columns and not args.skip_label_check:
        redo = [label_correct(a, r) for a, r in zip(src[acol], src[rcol])]
        agree = sum(int(x == int(y)) for x, y in zip(redo, src[TARGET])) / len(src)
        print(f"Label self-check vs V3 ({acol!r}): {agree:.1%} agreement")
        if agree < 0.95:
            raise SystemExit(
                "label_correct() does not reproduce the V3 labels. Replace its body with the "
                "matching rule from your original benchmark_300.py (or pass --skip-label-check)."
            )
    else:
        print("Label self-check skipped (no V3 answer/label columns found or flag set).")

    fingerprint = config_fingerprint()
    done: set[int] = set()
    if output.exists():
        prev = pd.read_csv(output)
        if set(prev["generation_fingerprint"].unique()) != {fingerprint}:
            raise SystemExit("Existing output was made with a different config. Use a new --output.")
        done = set(prev["row_id"].astype(int))
        print(f"Resuming: {len(done)} rows already done.")

    todo = [i for i in range(len(src) if args.limit is None else min(args.limit, len(src))) if i not in done]
    print(f"Prompt {PROMPT_VERSION} | fingerprint {fingerprint} | {len(todo)} questions to run")
    if not todo:
        return

    import torch

    llm = load_local_llm(model_name=MODEL_NAME, prefer_gpu=True)
    started = perf_counter()

    for n, i in enumerate(todo, start=1):
        row = src.iloc[i]
        question = str(row[qcol]).strip()
        t0 = perf_counter()
        torch.manual_seed(args.seed + i)

        results = sample_answers(llm, question)
        features = ReliabilityGate._build_features(question, results)
        answers = [r.answer.strip() for r in results]
        consensus = ReliabilityGate._consensus_answer(answers, question)

        record = {
            "row_id": i,
            **({"question_id": row["question_id"]} if "question_id" in src.columns else {}),
            "question": question,
            "reference": row[rcol],
            "consensus_answer": consensus,
            TARGET: label_correct(consensus, row[rcol]),
            **features,
            **{f"sample_{k + 1}": a for k, a in enumerate(answers)},
            "prompt_version": PROMPT_VERSION,
            "generation_fingerprint": fingerprint,
            "latency_seconds": round(perf_counter() - t0, 2),
        }
        pd.DataFrame([record]).to_csv(output, mode="a", header=not output.exists(), index=False, encoding="utf-8")

        elapsed = perf_counter() - started
        eta = elapsed / n * (len(todo) - n) / 60
        print(f"[{n}/{len(todo)}] id={i} correct={record[TARGET]} agree={features['answer_agreement']:.1f} ETA {eta:.0f} min")

    out = pd.read_csv(output)
    print(f"\nSaved {output}\nRows {len(out)} | correct {int(out[TARGET].sum())} | wrong {int((1 - out[TARGET]).sum())} | mean truncation_rate {out['truncation_rate'].mean():.3f}")


if __name__ == "__main__":
    main()
