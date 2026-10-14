"""Train + calibrate the deployable V4 gate.

Threshold comes ONLY from selected_threshold_v4.json (written by
risk_coverage_analysis.py from train-only OOF predictions). Nothing here looks at
test performance. V3 artifacts are never touched.

    python scripts\\finalize_gate_v4.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.config import (
    CLASSIFIER_DIR, EXPECTED_FEATURES, GATE_FILE_V4, PROMPT_VERSION,
    config_fingerprint, generation_config,
)

TARGET = "consensus_correct"
MIN_SEEDS_MEETING_TARGET = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmark_results_300_v4.csv")
    ap.add_argument("--threshold-json", default="selected_threshold_v4.json")
    ap.add_argument("--allow-unmet", action="store_true",
                    help="Deploy even if the OOF trusted-accuracy target was not met on most seeds.")
    args = ap.parse_args()

    input_file = ROOT_DIR / "artifacts" / "results" / args.input
    thr_file = CLASSIFIER_DIR / args.threshold_json
    for f in (input_file, thr_file):
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}. Run the earlier workflow steps first.")

    df = pd.read_csv(input_file)

    fingerprint = config_fingerprint()
    if set(df["generation_fingerprint"].unique()) != {fingerprint}:
        raise SystemExit(
            "Benchmark was generated with a different prompt/generation config than the "
            "current code. Regenerate it (scripts\\benchmark_300_v4.py)."
        )

    missing = [c for c in EXPECTED_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    df = df.dropna(subset=EXPECTED_FEATURES + [TARGET]).copy()
    X, y = df[EXPECTED_FEATURES].astype(float), df[TARGET].astype(int)

    thr = json.loads(thr_file.read_text(encoding="utf-8"))
    threshold = float(thr["threshold"])
    met = int(thr["seeds_meeting_oof_target"])
    if met < MIN_SEEDS_MEETING_TARGET and not args.allow_unmet:
        raise SystemExit(
            f"Only {met}/{len(thr['seeds'])} seeds reached the {thr['min_trusted_accuracy_target']:.0%} "
            "trusted-accuracy target on train-OOF predictions, so this threshold is a fallback, not a "
            "validated operating point. Inspect risk_coverage_summary_v4.txt, or pass --allow-unmet."
        )

    base = ExtraTreesClassifier(n_estimators=500, max_depth=7, min_samples_leaf=3,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5, n_jobs=-1)
    model.fit(X, y)

    payload = {
        "model": model,
        "feature_names": list(EXPECTED_FEATURES),
        "trust_threshold": threshold,
        "target_name": TARGET,
        "model_type": "ExtraTreesClassifier + Sigmoid Calibration",
        "dataset_rows": int(len(df)),
        "correct_rows": int((y == 1).sum()),
        "wrong_rows": int((y == 0).sum()),
        "threshold_source": thr["selection_rule"],
        "threshold_details": thr,
        "benchmark_file": str(input_file),
        "version": f"final_v4_{PROMPT_VERSION}",
        "prompt_version": PROMPT_VERSION,
        "generation_config": generation_config(),
        "generation_fingerprint": fingerprint,
    }
    joblib.dump(payload, GATE_FILE_V4)

    # Reload check
    again = joblib.load(GATE_FILE_V4)
    p = again["model"].predict_proba(X.head(3))[:, 1]

    cfg = generation_config()
    meta = f"""LOCAL LLM RELIABILITY GATE - FINAL MODEL (V4)

Prompt version: {PROMPT_VERSION}
Generation fingerprint: {fingerprint}
Model: {cfg['model_name']} | samples={cfg['n_samples']} | max_new_tokens={cfg['max_new_tokens']}
Sampling: temperature={cfg['temperature']} top_p={cfg['top_p']} top_k={cfg['top_k']}

Classifier: ExtraTreesClassifier + sigmoid calibration
Training rows: {len(df)} (correct {(y == 1).sum()}, wrong {(y == 0).sum()})
Benchmark file: {input_file.name}

Features ({len(EXPECTED_FEATURES)}):
{chr(10).join('- ' + f for f in EXPECTED_FEATURES)}

Deployment threshold: {threshold:.2f}
Source: {thr['selection_rule']}
Per-seed OOF thresholds: {thr['per_seed_thresholds']}
Seeds meeting OOF trusted-accuracy target: {met}/{len(thr['seeds'])}

Decision: P(correct) >= {threshold:.2f} -> TRUST, otherwise VERIFY.
Threshold was chosen from train-OOF predictions only, never from test results.
This is a reliability/abstention gate, not a truth detector.
"""
    (CLASSIFIER_DIR / "final_gate_metadata_v4.txt").write_text(meta, encoding="utf-8")

    print(f"Saved {GATE_FILE_V4}")
    print(f"Threshold {threshold:.2f} | prompt {PROMPT_VERSION} | fingerprint {fingerprint}")
    print(f"Reload check OK (sample P(correct): {[round(float(v), 3) for v in p]})")


if __name__ == "__main__":
    main()
