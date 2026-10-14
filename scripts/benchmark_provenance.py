"""Verify benchmark question/reference rows against the canonical local source."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CANONICAL_DATASET = ROOT_DIR / "data" / "sample" / "benchmark_questions_300.csv"


def audit_reference_provenance(df) -> tuple[str, list[str]]:
    """Return the source SHA-256 and any row-level question/reference mismatches."""
    digest = hashlib.sha256(CANONICAL_DATASET.read_bytes()).hexdigest()
    with CANONICAL_DATASET.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        canonical_rows = list(reader)

    canonical = {row["id"]: row for row in canonical_rows}
    mismatches: list[str] = []
    if len(canonical_rows) != 300 or len(canonical) != 300:
        mismatches.append(f"canonical dataset has {len(canonical_rows)} rows and {len(canonical)} unique IDs")
    if "question_id" not in df.columns:
        return digest, [*mismatches, "benchmark is missing question_id"]

    ids = df["question_id"].astype(str)
    if ids.duplicated().any() or set(ids) != set(canonical):
        mismatches.append("benchmark question_id values do not exactly cover the canonical source IDs")

    for row in df.itertuples(index=False):
        row_id = str(getattr(row, "question_id"))
        expected = canonical.get(row_id)
        if expected is None:
            continue
        if str(getattr(row, "question")) != expected["question"]:
            mismatches.append(f"{row_id}: question differs from canonical source")
        if str(getattr(row, "reference")) != expected["expected"]:
            mismatches.append(f"{row_id}: reference differs from canonical expected answer")
    return digest, mismatches
