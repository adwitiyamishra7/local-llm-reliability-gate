"""
V5 post-benchmark workflow: audit reference labels, evaluate, and finalize the gate.

Run after benchmark_300_v4.py --output benchmark_results_v5.csv has completed:
    python scripts/run_v5_training.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 65}")
    print(f"STEP: {label}")
    print("=" * 65)
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {label} (exit code {result.returncode})")
    else:
        print(f"\n[OK] {label}")
    return result.returncode


def main() -> int:
    steps = [
        (
            [PYTHON, "scripts/relabel_v5_benchmark.py",
             "--input", "benchmark_results_v5.csv",
             "--output", "benchmark_results_v5_labeled.csv"],
            "Audit labels against stored references (preserving the raw benchmark)",
        ),
        (
            [PYTHON, "scripts/risk_coverage_analysis.py",
             "--input", "benchmark_results_v5_labeled.csv", "--tag", "v5"],
            "Risk-coverage analysis (produces selected_threshold_v5.json)",
        ),
        (
            [PYTHON, "scripts/finalize_gate_v5.py",
             "--input", "benchmark_results_v5_labeled.csv",
             "--threshold-json", "selected_threshold_v5.json"],
            "Finalize V5 gate (produces reliability_gate_final_v5.pkl)",
        ),
        (
            [PYTHON, "scripts/test_v5_artifact.py"],
            "Artifact integration test",
        ),
        (
            [PYTHON, "-m", "pytest", "-q", "--tb=short"],
            "Full test suite",
        ),
        (
            [PYTHON, "scripts/smoke_test_verifier.py"],
            "Live verifier smoke test",
        ),
    ]

    for cmd, label in steps:
        code = run(cmd, label)
        if code != 0:
            print(f"\n[ABORT] Stopping at failed step: {label}")
            return code

    print("\n" + "=" * 65)
    print("V5 TRAINING WORKFLOW COMPLETE")
    print("=" * 65)
    print("All steps passed. The V5 gate is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
