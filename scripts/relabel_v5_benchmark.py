"""Create a label-audited V5 dataset without modifying generated benchmark rows.

The raw benchmark is preserved exactly as generated. This command writes a
separate CSV whose target is recomputed only from each stored curated reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.benchmark_300_v4 import label_correct
from scripts.benchmark_provenance import audit_reference_provenance
from src.gate.config import PROMPT_VERSION, config_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="benchmark_results_v5.csv")
    parser.add_argument("--output", default="benchmark_results_v5_labeled.csv")
    args = parser.parse_args()

    results_dir = ROOT_DIR / "artifacts" / "results"
    source = (results_dir / args.input).resolve()
    destination = (results_dir / args.output).resolve()
    if source == destination:
        raise SystemExit("Refusing to overwrite the generated benchmark.")
    df = pd.read_csv(source)
    required = {
        "row_id", "question_id", "question", "reference", "consensus_answer",
        "prompt_version", "generation_fingerprint",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing benchmark provenance columns: {missing}")
    ids = pd.to_numeric(df["row_id"], errors="coerce")
    if (
        len(df) != 300
        or ids.isna().any()
        or ids.duplicated().any()
        or set(ids.astype(int)) != set(range(300))
    ):
        raise SystemExit("Label audit requires exactly one row for each benchmark ID 0..299.")
    if set(df["generation_fingerprint"].astype(str)) != {config_fingerprint()}:
        raise SystemExit("Generation fingerprint does not match the live V5 configuration.")
    if set(df["prompt_version"].astype(str)) != {PROMPT_VERSION}:
        raise SystemExit("Prompt version does not match the live V5 configuration.")
    if df[["question", "reference", "consensus_answer"]].isna().any().any():
        raise SystemExit("Question, reference, and consensus answer fields must be present.")
    source_hash, provenance_errors = audit_reference_provenance(df)
    if provenance_errors:
        raise SystemExit("Canonical question/reference provenance failed: " + "; ".join(provenance_errors[:5]))

    df["consensus_correct"] = [
        label_correct(answer, reference)
        for answer, reference in zip(df["consensus_answer"], df["reference"])
    ]
    df["label_method"] = "reference_alias_exact_v2"
    df["reference_dataset_sha256"] = source_hash
    if destination.exists():
        existing = pd.read_csv(destination)
        if existing.equals(df):
            print(f"Audited benchmark already exists and matches: {destination}")
            return 0
        raise SystemExit(f"Refusing to overwrite existing audited data: {destination}")
    df.to_csv(destination, index=False, encoding="utf-8")
    counts = df["consensus_correct"].value_counts().to_dict()
    print(f"Audited benchmark written to {destination}")
    print(f"Rows: {len(df)} | correct: {counts.get(1, 0)} | wrong: {counts.get(0, 0)}")
    print("Source raw benchmark remains unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
