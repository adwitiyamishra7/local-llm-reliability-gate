from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    / "benchmark_results_300_v3.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "artifacts"
    / "classifier"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "reliability_gate_v2.pkl"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "training_metrics_v2.txt"
)


# ============================================================
# FEATURE SET
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


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark result file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = FEATURES + [TARGET]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    df = df[required].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required
    )

    df[TARGET] = (
        df[TARGET]
        .astype(int)
    )

    return df


# ============================================================
# MODELS
# ============================================================

def build_models():

    models = {}

    models["LogisticRegression"] = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    models["HistGradientBoosting"] = (
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )
    )

    return models


# ============================================================
# SELECT THRESHOLD
# ============================================================

def choose_reliability_threshold(
    y_true,
    probabilities,
):

    candidates = np.arange(
        0.50,
        0.96,
        0.01,
    )

    best_threshold = 0.50
    best_score = -1.0

    for threshold in candidates:

        predicted = (
            probabilities >= threshold
        ).astype(int)

        trusted = (
            predicted == 1
        )

        coverage = trusted.mean()

        if coverage == 0:
            continue

        trusted_accuracy = (
            y_true[trusted].mean()
        )

        # Prefer high precision while
        # retaining meaningful coverage.
        if trusted_accuracy >= 0.90:

            score = (
                trusted_accuracy
                * coverage
            )

            if score > best_score:

                best_score = score
                best_threshold = threshold

    return best_threshold


# ============================================================
# SELECTIVE EVALUATION
# ============================================================

def selective_metrics(
    y_true,
    probabilities,
    threshold,
):

    trusted = (
        probabilities >= threshold
    )

    coverage = (
        trusted.mean()
    )

    if trusted.sum() > 0:

        trusted_accuracy = (
            y_true[trusted].mean()
        )

    else:

        trusted_accuracy = 0.0

    trusted_count = int(
        trusted.sum()
    )

    verify_count = (
        len(y_true)
        - trusted_count
    )

    return (
        coverage,
        trusted_accuracy,
        trusted_count,
        verify_count,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOCAL LLM RELIABILITY GATE")
    print("RELIABILITY GATE V2 TRAINING")
    print("=" * 70)

    df = load_dataset()

    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    print()
    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Feature count: {len(FEATURES)}"
    )

    print()
    print("Class distribution:")

    print(
        f"Correct: {(y == 1).sum()}"
    )

    print(
        f"Wrong: {(y == 0).sum()}"
    )

    # ========================================================
    # FINAL HOLDOUT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=42,
        )
    )

    print()
    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Final test rows: {len(X_test)}"
    )

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = build_models()

    scores = {}

    print()
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for name, model in models.items():

        accuracy = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
        )

        roc_auc = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
        )

        f1 = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="f1",
        )

        scores[name] = {
            "accuracy": accuracy.mean(),
            "accuracy_std": accuracy.std(),
            "roc_auc": roc_auc.mean(),
            "f1": f1.mean(),
        }

        print()
        print(name)

        print(
            f"CV Accuracy : "
            f"{accuracy.mean():.4f}"
            f" ± {accuracy.std():.4f}"
        )

        print(
            f"CV ROC-AUC  : "
            f"{roc_auc.mean():.4f}"
        )

        print(
            f"CV F1       : "
            f"{f1.mean():.4f}"
        )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_name = max(
        scores,
        key=lambda name: scores[name]["roc_auc"],
    )

    best_base_model = models[
        best_name
    ]

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    print(
        f"Selected: {best_name}"
    )

    # ========================================================
    # OOF PROBABILITIES FOR THRESHOLD
    # ========================================================

    oof_probabilities = cross_val_predict(
        best_base_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    threshold = (
        choose_reliability_threshold(
            y_train.to_numpy(),
            oof_probabilities,
        )
    )

    print()
    print(
        f"Selected reliability threshold: "
        f"{threshold:.2f}"
    )

    # ========================================================
    # CALIBRATION
    # ========================================================

    calibrated_model = (
        CalibratedClassifierCV(
            estimator=best_base_model,
            method="sigmoid",
            cv=5,
        )
    )

    print()
    print(
        "Training calibrated reliability gate..."
    )

    calibrated_model.fit(
        X_train,
        y_train,
    )

    print(
        "Calibration complete."
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    y_probability = (
        calibrated_model
        .predict_proba(X_test)[:, 1]
    )

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_prediction,
    )

    precision = precision_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_prediction,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    brier = brier_score_loss(
        y_test,
        y_probability,
    )

    matrix = confusion_matrix(
        y_test,
        y_prediction,
    )

    (
        coverage,
        trusted_accuracy,
        trusted_count,
        verify_count,
    ) = selective_metrics(
        y_test.to_numpy(),
        y_probability,
        threshold,
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL HOLDOUT TEST")
    print("=" * 70)

    print()
    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Precision       : {precision:.4f}"
    )

    print(
        f"Recall          : {recall:.4f}"
    )

    print(
        f"F1              : {f1:.4f}"
    )

    print(
        f"ROC-AUC         : {roc_auc:.4f}"
    )

    print(
        f"Brier score     : {brier:.4f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(matrix)

    print()
    print(
        "Classification report:"
    )

    print(
        classification_report(
            y_test,
            y_prediction,
            target_names=[
                "Wrong",
                "Correct",
            ],
            zero_division=0,
        )
    )

    # ========================================================
    # SELECTIVE RELIABILITY
    # ========================================================

    print("=" * 70)
    print("RELIABILITY GATE BEHAVIOR")
    print("=" * 70)

    print()
    print(
        f"Threshold       : {threshold:.2f}"
    )

    print(
        f"Trusted answers : "
        f"{trusted_count}/{len(y_test)}"
    )

    print(
        f"Coverage        : "
        f"{coverage:.2%}"
    )

    print(
        f"Trusted accuracy: "
        f"{trusted_accuracy:.2%}"
    )

    print(
        f"Verify answers  : "
        f"{verify_count}/{len(y_test)}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": calibrated_model,
        "features": FEATURES,
        "threshold": threshold,
        "selected_base_model": best_name,
    }

    with MODEL_FILE.open(
        "wb"
    ) as file:

        pickle.dump(
            payload,
            file,
        )

    # ========================================================
    # METRICS FILE
    # ========================================================

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "LOCAL LLM RELIABILITY GATE V2\n"
        )

        file.write(
            "=" * 60
            + "\n"
        )

        file.write(
            f"Dataset rows: {len(df)}\n"
        )

        file.write(
            f"Training rows: {len(X_train)}\n"
        )

        file.write(
            f"Final test rows: {len(X_test)}\n"
        )

        file.write(
            f"Selected model: {best_name}\n"
        )

        file.write(
            f"Threshold: {threshold:.6f}\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F1: {f1:.6f}\n"
        )

        file.write(
            f"ROC-AUC: {roc_auc:.6f}\n"
        )

        file.write(
            f"Brier: {brier:.6f}\n"
        )

        file.write(
            f"Coverage: {coverage:.6f}\n"
        )

        file.write(
            f"Trusted accuracy: "
            f"{trusted_accuracy:.6f}\n"
        )

        file.write(
            "\nFeatures:\n"
        )

        for feature in FEATURES:

            file.write(
                f"- {feature}\n"
            )

        file.write(
            "\nCross-validation:\n"
        )

        for name, result in scores.items():

            file.write(
                f"\n{name}\n"
            )

            file.write(
                f"Accuracy: "
                f"{result['accuracy']:.6f}\n"
            )

            file.write(
                f"Accuracy std: "
                f"{result['accuracy_std']:.6f}\n"
            )

            file.write(
                f"ROC-AUC: "
                f"{result['roc_auc']:.6f}\n"
            )

            file.write(
                f"F1: "
                f"{result['f1']:.6f}\n"
            )

    print()
    print("=" * 70)
    print("RELIABILITY GATE V2 SAVED")
    print("=" * 70)

    print()
    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )


if __name__ == "__main__":
    main()