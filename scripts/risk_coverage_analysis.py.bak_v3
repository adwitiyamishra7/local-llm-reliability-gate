from __future__ import annotations

"""
Risk-Coverage + Threshold Analysis for Local LLM Reliability Gate.

Purpose
-------
Evaluate whether the reliability gate is genuinely conservative/reliable
across multiple train/test splits, without using the final test set to choose
an operating threshold.

Inputs
------
artifacts/results/benchmark_results_300_v3.csv

Outputs
-------
artifacts/classifier/risk_coverage_thresholds.csv
artifacts/classifier/risk_coverage_summary.txt
artifacts/classifier/risk_coverage_curve.png
artifacts/classifier/test_predictions_seed42.csv

Protocol per seed
-----------------
1. Stratified 80/20 train/test split.
2. Use the current best V2 model family: ExtraTrees.
3. Generate out-of-fold calibrated probabilities on TRAIN only.
4. Select the operating threshold from OOF predictions, never from TEST.
5. Fit the calibrated gate on the full TRAIN split.
6. Evaluate every threshold on the untouched TEST split.

This prevents test-set leakage when choosing the deployment threshold.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT_DIR / "artifacts" / "results" / "benchmark_results_300_v3.csv"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.round(np.arange(0.50, 0.951, 0.01), 2)
SEEDS = [42, 52, 62, 72, 82]
TEST_SIZE = 0.20
RANDOM_STATE_BASE = 42

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
MIN_TRUSTED_ACCURACY = 0.90


def build_calibrated_model(seed: int) -> CalibratedClassifierCV:
    """Build the same ExtraTrees + sigmoid calibration family used in V2."""
    base_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )

    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
        n_jobs=-1,
    )


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Compute selective reliability metrics at one threshold."""
    trusted = probabilities >= threshold
    trusted_count = int(trusted.sum())
    total = len(y_true)
    coverage = trusted_count / total if total else 0.0

    if trusted_count > 0:
        trusted_accuracy = float(np.mean(y_true[trusted] == 1))
    else:
        trusted_accuracy = np.nan

    risk = float(1.0 - trusted_accuracy) if trusted_count > 0 else np.nan

    # This is not end-to-end system accuracy because VERIFY outcomes are not known.
    trusted_correct_rate_over_all = (
        float(np.sum(y_true[trusted] == 1)) / total if total else 0.0
    )

    return {
        "threshold": float(threshold),
        "trusted_count": trusted_count,
        "coverage": coverage,
        "trusted_accuracy": trusted_accuracy,
        "risk": risk,
        "trusted_correct_rate_over_all": trusted_correct_rate_over_all,
        "verify_count": int(total - trusted_count),
    }


def select_threshold_from_oof(y_true: np.ndarray, oof_probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Select a threshold using TRAIN-only OOF probabilities.

    Selection rule:
    - Require trusted accuracy >= 90% whenever possible.
    - Among qualifying thresholds, maximize coverage.
    - Tie-break with higher trusted accuracy, then lower threshold.
    """
    rows = [threshold_metrics(y_true, oof_probabilities, t) for t in THRESHOLDS]
    table = pd.DataFrame(rows)

    qualifying = table[
        table["trusted_accuracy"].notna()
        & (table["trusted_accuracy"] >= MIN_TRUSTED_ACCURACY)
    ]

    if not qualifying.empty:
        selected = qualifying.sort_values(
            ["coverage", "trusted_accuracy", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        return float(selected["threshold"]), table

    # Fallback: no threshold reaches the target reliability.
    fallback = table.sort_values(
        ["trusted_accuracy", "coverage", "threshold"],
        ascending=[False, False, True],
        na_position="last",
    ).iloc[0]
    return float(fallback["threshold"]), table


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return float("nan")


def run_one_seed(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Run one complete honest train/OOF-threshold/test evaluation."""
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=seed,
    )

    # OOF probabilities for TRAIN only. The threshold is selected from these.
    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_model = build_calibrated_model(seed)
    oof_probabilities = cross_val_predict(
        oof_model,
        X_train,
        y_train,
        cv=oof_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    selected_threshold, oof_table = select_threshold_from_oof(
        y_train.to_numpy(), oof_probabilities
    )

    # Final model is fit ONLY on the training split.
    final_model = build_calibrated_model(seed)
    final_model.fit(X_train, y_train)
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_rows = [
        {
            "seed": seed,
            **threshold_metrics(y_test.to_numpy(), test_probabilities, t),
        }
        for t in THRESHOLDS
    ]
    test_table = pd.DataFrame(test_rows)

    selected_test = threshold_metrics(
        y_test.to_numpy(), test_probabilities, selected_threshold
    )

    test_prediction_table = pd.DataFrame(
        {
            "row_index": idx_test,
            "consensus_correct": y_test.to_numpy(),
            "predicted_probability_correct": test_probabilities,
            "selected_threshold": selected_threshold,
            "decision": np.where(test_probabilities >= selected_threshold, "TRUST", "VERIFY"),
        }
    ).sort_values("row_index")

    summary = {
        "seed": seed,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "selected_threshold": selected_threshold,
        "test_roc_auc": safe_roc_auc(y_test.to_numpy(), test_probabilities),
        "test_brier": float(brier_score_loss(y_test, test_probabilities)),
        "test_accuracy_at_0_50": float(
            accuracy_score(y_test, (test_probabilities >= 0.50).astype(int))
        ),
        "selected_coverage": selected_test["coverage"],
        "selected_trusted_accuracy": selected_test["trusted_accuracy"],
        "selected_risk": selected_test["risk"],
        "selected_trusted_count": selected_test["trusted_count"],
        "selected_verify_count": selected_test["verify_count"],
    }

    # Preserve the OOF table too for debugging/inspection if needed.
    oof_table.insert(0, "seed", seed)

    return test_table, summary, test_prediction_table


def summarize_thresholds(all_threshold_results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        all_threshold_results.groupby("threshold", as_index=False)
        .agg(
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_trusted_accuracy=("trusted_accuracy", "mean"),
            std_trusted_accuracy=("trusted_accuracy", "std"),
            mean_risk=("risk", "mean"),
            std_risk=("risk", "std"),
            mean_trusted_correct_rate_over_all=("trusted_correct_rate_over_all", "mean"),
            mean_verify_count=("verify_count", "mean"),
            mean_trusted_count=("trusted_count", "mean"),
        )
        .sort_values("threshold")
    )
    return grouped


def choose_robust_threshold(split_summary_df: pd.DataFrame) -> tuple[float, str]:
    """
    Choose one deployment threshold using TRAIN-only decisions.

    Each seed already selected its threshold from TRAIN OOF predictions.
    The median of those thresholds is therefore a leakage-safe aggregate.
    We deliberately do NOT choose the deployment threshold from TEST results.
    """
    selected_thresholds = split_summary_df["selected_threshold"].dropna().to_numpy(dtype=float)
    if selected_thresholds.size == 0:
        raise ValueError("No per-seed OOF-selected thresholds were produced.")

    robust_threshold = float(np.median(selected_thresholds))
    return robust_threshold, "median of per-seed thresholds selected from TRAIN-only OOF predictions"


def save_curve(summary_table: pd.DataFrame, selected_threshold: float, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(
        summary_table["mean_coverage"],
        summary_table["mean_risk"],
        marker="o",
        markersize=3,
        linewidth=1.5,
    )

    selected = summary_table.loc[
        np.isclose(summary_table["threshold"].to_numpy(), selected_threshold)
    ]
    if not selected.empty:
        row = selected.iloc[0]
        plt.scatter([row["mean_coverage"]], [row["mean_risk"]], s=70)
        plt.annotate(
            f"t={selected_threshold:.2f}",
            (row["mean_coverage"], row["mean_risk"]),
            xytext=(8, 8),
            textcoords="offset points",
        )

    plt.xlabel("Coverage (fraction of answers trusted)")
    plt.ylabel("Risk (1 - trusted accuracy)")
    plt.title("Local LLM Reliability Gate: Risk-Coverage Curve")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    print("=" * 70)
    print("LOCAL LLM RELIABILITY GATE")
    print("RISK-COVERAGE + THRESHOLD ANALYSIS")
    print("=" * 70)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark results not found:\n{INPUT_CSV}\n\n"
            "Run the 300-question benchmark first, or place the completed CSV "
            "at this exact path."
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "benchmark_results_300_v3.csv is missing required columns:\n"
            + "\n".join(f"- {col}" for col in missing)
        )

    df = df.dropna(subset=required_columns).copy()

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    invalid_targets = sorted(set(df[TARGET].unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"TARGET must contain only 0/1, found: {invalid_targets}")

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable rows found. Need substantially more than this for repeated split analysis.")

    print(f"Dataset rows: {len(df)}")
    print(f"Feature count: {len(FEATURES)}")
    print("\nClass distribution:")
    print(f"Correct: {int((df[TARGET] == 1).sum())}")
    print(f"Wrong:   {int((df[TARGET] == 0).sum())}")
    print(f"\nSeeds: {SEEDS}")

    all_threshold_results = []
    split_summaries = []
    seed42_predictions = None

    for seed in SEEDS:
        print("\n" + "-" * 70)
        print(f"SEED {seed}")
        print("-" * 70)

        test_table, summary, predictions = run_one_seed(df, seed)
        all_threshold_results.append(test_table)
        split_summaries.append(summary)

        if seed == 42:
            seed42_predictions = predictions

        print(f"Selected threshold (TRAIN OOF): {summary['selected_threshold']:.2f}")
        print(f"Test ROC-AUC            : {summary['test_roc_auc']:.4f}")
        print(f"Test Brier              : {summary['test_brier']:.4f}")
        print(f"Selected coverage       : {summary['selected_coverage']:.2%}")
        print(f"Selected trusted acc.   : {summary['selected_trusted_accuracy']:.2%}")
        print(f"Selected risk           : {summary['selected_risk']:.2%}")

    all_threshold_results_df = pd.concat(all_threshold_results, ignore_index=True)
    threshold_summary = summarize_thresholds(all_threshold_results_df)
    split_summary_df = pd.DataFrame(split_summaries)

    robust_threshold, selection_reason = choose_robust_threshold(split_summary_df)
    robust_row = threshold_summary.loc[
        np.isclose(threshold_summary["threshold"].to_numpy(), robust_threshold)
    ].iloc[0]

    # Diagnostic only: this is the best threshold on TEST and must never be
    # used as the deployment threshold or quoted as an unbiased estimate.
    test_candidates = threshold_summary[threshold_summary["mean_trusted_accuracy"].notna()].copy()
    test_oracle = test_candidates.sort_values(
        ["mean_coverage", "mean_trusted_accuracy", "threshold"],
        ascending=[False, False, True],
    )
    test_oracle_90 = test_oracle[test_oracle["mean_trusted_accuracy"] >= MIN_TRUSTED_ACCURACY]
    if not test_oracle_90.empty:
        test_oracle_row = test_oracle_90.iloc[0]
    else:
        test_oracle_row = test_candidates.sort_values(
            ["mean_trusted_accuracy", "mean_coverage", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]

    # Canonical seed-42 prediction file gets the robust threshold for inspection.
    if seed42_predictions is not None:
        seed42_predictions = seed42_predictions.copy()
        seed42_predictions["robust_threshold"] = robust_threshold
        seed42_predictions["robust_decision"] = np.where(
            seed42_predictions["predicted_probability_correct"] >= robust_threshold,
            "TRUST",
            "VERIFY",
        )
        # Attach original row identifiers when available.
        if "question_id" in df.columns:
            seed42_predictions = seed42_predictions.merge(
                df[["question_id"]],
                left_on="row_index",
                right_index=True,
                how="left",
            )

    thresholds_path = OUTPUT_DIR / "risk_coverage_thresholds.csv"
    summaries_path = OUTPUT_DIR / "risk_coverage_split_summaries.csv"
    predictions_path = OUTPUT_DIR / "test_predictions_seed42.csv"
    summary_path = OUTPUT_DIR / "risk_coverage_summary.txt"
    curve_path = OUTPUT_DIR / "risk_coverage_curve.png"

    threshold_summary.to_csv(thresholds_path, index=False)
    split_summary_df.to_csv(summaries_path, index=False)
    if seed42_predictions is not None:
        seed42_predictions.to_csv(predictions_path, index=False)

    save_curve(threshold_summary, robust_threshold, curve_path)

    # Useful operating points for the report.
    operating_points = {}
    for t in [0.50, 0.60, 0.70, 0.74, 0.80, 0.85, 0.90]:
        row = threshold_summary.loc[
            np.isclose(threshold_summary["threshold"].to_numpy(), t)
        ]
        if not row.empty:
            operating_points[t] = row.iloc[0]

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("LOCAL LLM RELIABILITY GATE\n")
        f.write("RISK-COVERAGE + THRESHOLD ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Dataset rows: {len(df)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Test fraction per seed: {TEST_SIZE:.0%}\n")
        f.write(f"Trusted accuracy target: {MIN_TRUSTED_ACCURACY:.0%}\n\n")

        f.write("PER-SEED RESULTS\n")
        f.write("-" * 70 + "\n")
        for row in split_summaries:
            f.write(
                f"Seed {row['seed']}: "
                f"threshold={row['selected_threshold']:.2f}, "
                f"ROC-AUC={row['test_roc_auc']:.4f}, "
                f"coverage={row['selected_coverage']:.2%}, "
                f"trusted_accuracy={row['selected_trusted_accuracy']:.2%}, "
                f"risk={row['selected_risk']:.2%}\n"
            )

        f.write("\nROBUST OPERATING THRESHOLD\n")
        f.write("-" * 70 + "\n")
        f.write(f"Threshold: {robust_threshold:.2f}\n")
        f.write(f"Selection rule: {selection_reason}\n")
        f.write(f"Mean coverage on test splits: {robust_row['mean_coverage']:.2%}\n")
        f.write(f"Mean trusted accuracy on test splits: {robust_row['mean_trusted_accuracy']:.2%}\n")
        f.write(f"Mean risk on test splits: {robust_row['mean_risk']:.2%}\n")
        f.write(f"Mean verify count on test splits: {robust_row['mean_verify_count']:.2f}\n")
        f.write("\nTEST-SET ORACLE (DIAGNOSTIC ONLY)\n")
        f.write("-" * 70 + "\n")
        f.write(f"Threshold: {test_oracle_row['threshold']:.2f}\n")
        f.write(f"Mean coverage: {test_oracle_row['mean_coverage']:.2%}\n")
        f.write(f"Mean trusted accuracy: {test_oracle_row['mean_trusted_accuracy']:.2%}\n")
        f.write("Do not use this threshold as the deployment choice; it is selected using TEST outcomes.\n")

        f.write("\nOPERATING POINTS\n")
        f.write("-" * 70 + "\n")
        for t, row in operating_points.items():
            f.write(
                f"t={t:.2f}: coverage={row['mean_coverage']:.2%}, "
                f"trusted_accuracy={row['mean_trusted_accuracy']:.2%}, "
                f"risk={row['mean_risk']:.2%}\n"
            )

        f.write("\nOUTPUT FILES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Threshold table: {thresholds_path}\n")
        f.write(f"Split summaries: {summaries_path}\n")
        f.write(f"Seed-42 predictions: {predictions_path}\n")
        f.write(f"Risk-coverage curve: {curve_path}\n")

    print("\n" + "=" * 70)
    print("ROBUST OPERATING THRESHOLD")
    print("=" * 70)
    print(f"Threshold                : {robust_threshold:.2f}")
    print(f"Mean coverage on tests    : {robust_row['mean_coverage']:.2%}")
    print(f"Mean trusted accuracy     : {robust_row['mean_trusted_accuracy']:.2%}")
    print(f"Mean risk on tests        : {robust_row['mean_risk']:.2%}")
    print(f"Mean verify count         : {robust_row['mean_verify_count']:.2f}")
    print("Threshold was NOT selected from test results.")

    print("\nOPERATING POINTS")
    print("=" * 70)
    print("Threshold | Coverage | Trusted Accuracy | Risk")
    print("-" * 70)
    for t, row in operating_points.items():
        print(
            f"   {t:.2f}   |  {row['mean_coverage']:.2%} |"
            f"       {row['mean_trusted_accuracy']:.2%}      | {row['mean_risk']:.2%}"
        )

    print("\nFILES SAVED")
    print(f"Threshold table : {thresholds_path}")
    print(f"Split summaries : {summaries_path}")
    print(f"Seed-42 tests   : {predictions_path}")
    print(f"Risk curve      : {curve_path}")
    print(f"Summary         : {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
