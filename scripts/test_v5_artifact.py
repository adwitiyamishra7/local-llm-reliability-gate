"""
V5 artifact integration test: validate that the produced reliability_gate_final_v5.pkl
is compatible with the production gate, has the correct fingerprint, and that the
decision path behaves correctly (fail-closed when calibration is missing, VERIFY for
unsupported questions, etc.).

Run after finalize_gate_v5.py has produced the artifact:
    python scripts/test_v5_artifact.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import pandas as pd

from src.gate.config import (
    CLASSIFIER_DIR,
    EXPECTED_FEATURES,
    GATE_FILE_V5,
    PROMPT_VERSION,
    config_fingerprint,
    generation_config,
)
from src.gate.reliability_gate import ReliabilityGate
from scripts.benchmark_provenance import audit_reference_provenance


PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def run() -> int:
    print("=" * 65)
    print("V5 ARTIFACT INTEGRATION TEST")
    print("=" * 65)

    # ----------------------------------------------------------------
    # 1. Artifact exists
    print("\n--- 1. Artifact Existence ---")
    check("V5 artifact file exists", GATE_FILE_V5.exists(), str(GATE_FILE_V5))
    if not GATE_FILE_V5.exists():
        print("\nArtifact does not exist. Run finalize_gate_v5.py first.")
        return 1

    # ----------------------------------------------------------------
    # 2. Schema check
    print("\n--- 2. Schema Validation ---")
    payload = joblib.load(GATE_FILE_V5)
    check("Payload is dict", isinstance(payload, dict))
    required_keys = {
        "model", "feature_names", "trust_threshold", "generation_config",
        "generation_fingerprint", "dataset_rows", "benchmark_source_rows",
        "unique_question_rows", "duplicate_questions_excluded", "label_method",
        "heldout_question_rows", "training_question_ids", "heldout_question_ids",
        "reference_dataset_sha256", "threshold_details", "threshold_source",
    }
    missing_keys = required_keys - set(payload.keys())
    check("All required keys present", not missing_keys, f"missing={missing_keys}" if missing_keys else "")

    feature_names = payload.get("feature_names", [])
    check(
        "Feature names match EXPECTED_FEATURES",
        list(feature_names) == EXPECTED_FEATURES,
        f"saved={feature_names}" if list(feature_names) != EXPECTED_FEATURES else "",
    )

    threshold = payload.get("trust_threshold", None)
    threshold_ok = (
        threshold is not None
        and math.isfinite(float(threshold))
        and 0.0 <= float(threshold) <= 1.0
    )
    check("Trust threshold is a valid float in [0,1]", threshold_ok, f"threshold={threshold}")

    # ----------------------------------------------------------------
    # 3. Fingerprint check
    print("\n--- 3. Fingerprint Validation ---")
    live_fp = config_fingerprint()
    saved_fp = payload.get("generation_fingerprint", "MISSING")
    check(
        "Saved fingerprint matches live V5 config",
        str(saved_fp) == live_fp,
        f"saved={saved_fp!r}  live={live_fp!r}",
    )

    saved_gen_cfg = payload.get("generation_config", {})
    live_gen_cfg = generation_config()
    cfg_matches = saved_gen_cfg == live_gen_cfg
    check("generation_config dict matches live config", cfg_matches)
    if not cfg_matches:
        diffs = {k for k in set(saved_gen_cfg) | set(live_gen_cfg) if saved_gen_cfg.get(k) != live_gen_cfg.get(k)}
        print(f"      Differing keys: {diffs}")

    # ----------------------------------------------------------------
    # Benchmark and held-out provenance
    print("\n--- Benchmark and Held-Out Provenance ---")
    source_rows = payload.get("benchmark_source_rows")
    unique_rows = payload.get("unique_question_rows")
    dataset_rows = payload.get("dataset_rows")
    duplicate_rows = payload.get("duplicate_questions_excluded")
    heldout_rows = payload.get("heldout_question_rows")
    training_ids = set(payload.get("training_question_ids", []))
    heldout_ids = set(payload.get("heldout_question_ids", []))
    threshold_details_seeds = payload.get("threshold_details", {}).get("seeds", [])
    provenance_ok = (
        isinstance(source_rows, int)
        and source_rows == 300
        and isinstance(unique_rows, int)
        and isinstance(duplicate_rows, int)
        and source_rows - duplicate_rows == unique_rows
        and isinstance(dataset_rows, int)
        and dataset_rows == len(training_ids)
        and isinstance(heldout_rows, int)
        and heldout_rows == len(heldout_ids)
        and dataset_rows + heldout_rows == unique_rows
        and not (training_ids & heldout_ids)
        and len(threshold_details_seeds) == 1
        and threshold_details_seeds == [42]
    )
    check(
        "Training uses complete source rows and unique held-out questions",
        provenance_ok,
        f"source={source_rows}, train={dataset_rows}, heldout={heldout_rows}, "
        f"unique={unique_rows}, duplicates_excluded={duplicate_rows}",
    )
    check(
        "Training labels are reference-anchored, not agreement-anchored",
        payload.get("label_method") == "reference_alias_exact_v2",
        str(payload.get("label_method")),
    )
    audited_df = pd.read_csv(
        ROOT_DIR / "artifacts" / "results" / "benchmark_results_v5_labeled.csv"
    )
    source_hash, provenance_errors = audit_reference_provenance(audited_df)
    unique_questions = audited_df["question"].astype(str).str.casefold().str.replace(
        r"\s+", " ", regex=True
    ).str.strip()
    expected_question_ids = set(
        audited_df.loc[~unique_questions.duplicated(keep="first"), "question_id"].astype(str)
    )
    check(
        "Question/reference provenance matches canonical source",
        not provenance_errors
        and payload.get("reference_dataset_sha256") == source_hash
        and training_ids | heldout_ids == expected_question_ids,
        f"sha256={payload.get('reference_dataset_sha256')}, errors={provenance_errors[:3]}",
    )
    check(
        "Deployment threshold was selected from training-only OOF predictions",
        "TRAIN-only OOF" in str(payload.get("threshold_source", "")),
        str(payload.get("threshold_source", "")),
    )
    threshold_details = payload.get("threshold_details", {})
    heldout_accuracy = threshold_details.get("heldout_mean_trusted_accuracy", float("nan"))
    heldout_count = threshold_details.get("heldout_mean_trusted_count", 0)
    try:
        heldout_ok = (
            math.isfinite(float(heldout_accuracy))
            and float(heldout_accuracy) >= float(threshold_details.get("min_trusted_accuracy_target", 0.90))
            and float(heldout_count) > 0
            and float(threshold_details.get("heldout_min_trusted_accuracy", float("nan")))
            >= float(threshold_details.get("min_trusted_accuracy_target", 0.90))
            and int(threshold_details.get("heldout_min_trusted_count", 0)) > 0
            and threshold_details.get("heldout_all_seeds_meet_target") is True
            and threshold_details.get("generation_fingerprint") == live_fp
        )
    except (TypeError, ValueError):
        heldout_ok = False
    check(
        "Held-out acceptance metrics meet the recorded target and fingerprint",
        heldout_ok,
        f"accuracy={heldout_accuracy}, mean_trusted={heldout_count}",
    )

    # ----------------------------------------------------------------
    # 4. Calibration check (via ReliabilityGate._check_calibration)
    print("\n--- 4. Calibration Check ---")
    calibrated, note = ReliabilityGate._check_calibration(payload)
    check("Artifact passes calibration check", calibrated, note if not calibrated else "")

    # ----------------------------------------------------------------
    # 5. Model predict_proba sanity
    print("\n--- 5. Model Predict-Proba Sanity ---")
    model = payload.get("model")
    X_dummy = pd.DataFrame(
        [[float(i) for i in range(len(EXPECTED_FEATURES))]],
        columns=EXPECTED_FEATURES,
    )
    try:
        probs = model.predict_proba(X_dummy)
        p = float(probs[0, 1])
        proba_ok = math.isfinite(p) and 0.0 <= p <= 1.0
        check("predict_proba returns finite probability in [0,1]", proba_ok, f"p={p:.4f}")
    except Exception as exc:
        check("predict_proba call succeeds", False, str(exc))

    # ----------------------------------------------------------------
    # 6. Stale artifact test — make sure a fake stale artifact fails
    print("\n--- 6. Stale Artifact Rejection ---")
    stale_payload = {"generation_config": generation_config(), "generation_fingerprint": "stale-fp-000"}
    stale_ok, stale_note = ReliabilityGate._check_calibration(stale_payload)
    check("Stale fingerprint is rejected (not calibrated)", not stale_ok, stale_note if not stale_ok else "")

    # ----------------------------------------------------------------
    # Summary
    print("\n" + "=" * 65)
    passed = sum(1 for _, s, _ in results if s == PASS)
    total = len(results)
    print(f"V5 ARTIFACT INTEGRATION: {passed}/{total} checks passed")
    print("=" * 65)

    failed = [name for name, s, _ in results if s == FAIL]
    if failed:
        print("\nFAILED checks:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print("\nAll checks PASSED. V5 artifact is compatible with the production gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
