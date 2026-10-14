from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
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
    / "benchmark_results_50_clean.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "artifacts"
    / "classifier"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "reliability_gate.pkl"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "training_metrics.txt"
)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "mean_logprob_avg",
    "min_logprob_avg",
    "clean_answer_agreement",
]

TARGET = "clean_consensus_correct"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean benchmark file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = (
        FEATURES
        + [TARGET]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )

    df = df[
        required_columns
    ].copy()

    # Remove invalid rows
    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=required_columns
    )

    return df


# ============================================================
# TRAIN
# ============================================================

def main():

    print("=" * 65)
    print("LOCAL LLM RELIABILITY GATE")
    print("LOGISTIC REGRESSION TRAINING")
    print("=" * 65)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print()
    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Features: {', '.join(FEATURES)}"
    )

    print()

    class_counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    print("Class distribution:")

    print(
        f"Correct (1): "
        f"{class_counts.get(1, 0)}"
    )

    print(
        f"Wrong (0): "
        f"{class_counts.get(0, 0)}"
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X = df[FEATURES].astype(float)

    y = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )
    )

    print()
    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    print()
    print("Training Logistic Regression...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    if len(
        np.unique(y_test)
    ) == 2:

        roc_auc = roc_auc_score(
            y_test,
            y_prob,
        )

    else:

        roc_auc = float("nan")

    matrix = confusion_matrix(
        y_test,
        y_pred,
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if roc_auc == roc_auc:

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    print()
    print("Confusion matrix:")

    print(matrix)

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Wrong",
                "Correct",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Example probabilities
    # --------------------------------------------------------

    print("=" * 65)
    print("TEST PREDICTIONS")
    print("=" * 65)

    test_output = X_test.copy()

    test_output[
        "actual"
    ] = y_test.values

    test_output[
        "predicted"
    ] = y_pred

    test_output[
        "reliability_probability"
    ] = y_prob

    test_output = test_output.sort_index()

    for index, row in test_output.iterrows():

        print(
            f"\nRow {index}: "
            f"P(correct) = "
            f"{row['reliability_probability']:.3f}"
        )

        print(
            f"Actual: "
            f"{'CORRECT' if row['actual'] == 1 else 'WRONG'}"
        )

        print(
            f"Predicted: "
            f"{'CORRECT' if row['predicted'] == 1 else 'WRONG'}"
        )

    # --------------------------------------------------------
    # Model coefficients
    # --------------------------------------------------------

    classifier = model.named_steps[
        "classifier"
    ]

    print()
    print("=" * 65)
    print("MODEL COEFFICIENTS")
    print("=" * 65)

    for feature, coefficient in zip(
        FEATURES,
        classifier.coef_[0],
    ):

        print(
            f"{feature}: "
            f"{coefficient:.6f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_FILE.open(
        "wb"
    ) as file:

        pickle.dump(
            model,
            file,
        )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "LOCAL LLM RELIABILITY GATE\n"
        )

        file.write(
            "LOGISTIC REGRESSION TRAINING\n"
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
            f"Testing rows: {len(X_test)}\n"
        )

        file.write(
            f"Correct: {class_counts.get(1, 0)}\n"
        )

        file.write(
            f"Wrong: {class_counts.get(0, 0)}\n"
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
            "\nFeatures:\n"
        )

        for feature in FEATURES:

            file.write(
                f"- {feature}\n"
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("MODEL SAVED")
    print("=" * 65)

    print()
    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )


if __name__ == "__main__":
    main()