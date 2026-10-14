"""
LOCAL LLM RELIABILITY GATE
FINAL MODEL FREEZE

Purpose:
- Reuse the completed 300-question benchmark.
- Freeze the ExtraTrees model selected during V2 analysis.
- Use the robust operating threshold found by multi-seed analysis: 0.82.
- Train/calibrate the final deployable model on all 300 labeled benchmark rows.
- Save feature names + threshold + metadata together with the model.

No Qwen inference is performed here.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT_DIR / "artifacts" / "results" / "benchmark_results_300_v3.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "classifier"
OUTPUT_MODEL = OUTPUT_DIR / "reliability_gate_final.pkl"
OUTPUT_METADATA = OUTPUT_DIR / "final_gate_metadata.txt"


# ============================================================
# CONFIG
# ============================================================

FEATURES = [
    "mean_logprob_avg",
    "mean_logprob_std",
    "mean_logprob_best",
    "mean_logprob_worst",
    "min_logprob_avg",
    "min_logprob_std",
    "min_logprob_best",
    "min_logprob_worst",
    "logprob_confidence_range",
    "answer_agreement",
    "consensus_margin",
    "answer_key_entropy",
    "answer_length_avg",
    "answer_length_std",
    "truncation_rate",
]

TARGET = "consensus_correct"

# Chosen from the multi-seed risk-coverage analysis.
# This is a deployment policy threshold, not a test-set optimized threshold.
TRUST_THRESHOLD = 0.82


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("LOCAL LLM RELIABILITY GATE")
    print("FINAL MODEL FREEZE")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Missing required features:\n"
            + "\n".join(f"- {c}" for c in missing_features)
        )

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print(f"\nDataset rows used : {len(df)}")
    print(f"Feature count     : {len(FEATURES)}")
    print(f"Correct           : {(y == 1).sum()}")
    print(f"Wrong             : {(y == 0).sum()}")
    print(f"Trust threshold   : {TRUST_THRESHOLD:.2f}")

    # Same ExtraTrees family/hyperparameters selected in V2 CV.
    base_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining calibrated ExtraTrees...")
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
        n_jobs=-1,
    )
    calibrated_model.fit(X, y)

    payload = {
        "model": calibrated_model,
        "feature_names": FEATURES,
        "trust_threshold": TRUST_THRESHOLD,
        "target_name": TARGET,
        "model_type": "ExtraTreesClassifier + Sigmoid Calibration",
        "dataset_rows": int(len(df)),
        "correct_rows": int((y == 1).sum()),
        "wrong_rows": int((y == 0).sum()),
        "threshold_source": "Median of train-OOF-selected thresholds across seeds [42, 52, 62, 72, 82]",
        "benchmark_file": str(INPUT_FILE),
        "version": "final_v1",
    }

    joblib.dump(payload, OUTPUT_MODEL)

    metadata = f"""
LOCAL LLM RELIABILITY GATE - FINAL MODEL

Model:
ExtraTreesClassifier + sigmoid calibration

Training data:
Rows: {len(df)}
Correct: {(y == 1).sum()}
Wrong: {(y == 0).sum()}

Features ({len(FEATURES)}):
{chr(10).join("- " + f for f in FEATURES)}

Deployment threshold:
{TRUST_THRESHOLD:.2f}

Decision:
P(correct) >= {TRUST_THRESHOLD:.2f} -> TRUST
P(correct) <  {TRUST_THRESHOLD:.2f} -> VERIFY

Threshold source:
Median of train-OOF-selected thresholds across seeds
[42, 52, 62, 72, 82]

Important:
The benchmark's holdout results remain evaluation results.
This final model is retrained on all labeled benchmark rows only
after the operating threshold was fixed.

This is a reliability/abstention gate, not a truth detector.
"""
    OUTPUT_METADATA.write_text(metadata.strip() + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("FINAL MODEL SAVED")
    print("=" * 70)
    print(f"Model    : {OUTPUT_MODEL}")
    print(f"Metadata : {OUTPUT_METADATA}")
    print("\nDecision policy:")
    print(f"P(correct) >= {TRUST_THRESHOLD:.2f}  -> TRUST")
    print(f"P(correct) <  {TRUST_THRESHOLD:.2f}  -> VERIFY")
    print("\nNo LLM generation was performed.")


if __name__ == "__main__":
    main()
