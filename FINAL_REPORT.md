# Local LLM Reliability Gate — Finalization Report

Date: 2026-10-14

## Outcome

The Rahul Gandhi behavior was correct end to end. The gate does not claim to verify arbitrary biographical facts. The question returns `UNSUPPORTED`, the final decision is `VERIFY`, and the reliability estimate is not reported. The captured BJP hallucination is now a hermetic regression fixture, so it cannot later become `TRUST` merely because the classifier score is high.

The local V5 artifact is compatible with the live configuration on this machine. Its saved fingerprint and the live fingerprint are both `763d5cfaf7b17341`. The artifact is the local ignored file `artifacts/classifier/reliability_gate_final_v5.pkl`; generated model artifacts remain machine-local by design.

## What changed

- Created the rollback baseline commit `097772b` and tag `pre-finalization-baseline` before edits.
- Created cleanup commit `f21fa9e`: removed inert backup trees, `v4_repair`, the applied one-off patch, dead `configs/config.yaml`, stale setup/history files, and the duplicate bundle script; moved `.streamlit/config.toml` to the project root and retained one portable bundle script.
- Reconciled the documentation into `README.md`, `ARCHITECTURE.md`, and `CHANGELOG.md`.
- Added an argument-driven real-model diagnostic in `scripts/test_live_gate.py`. It prints `N/A` when reliability is not reportable and labels the raw classifier value as diagnostic only.
- Added a hermetic Rahul Gandhi regression in `tests/test_gate_v5.py` covering the exact captured BJP answer, `UNSUPPORTED`, `VERIFY`, no reported reliability, and the UI's `N/A`/unsupported presentation.
- Hardened `app.build_view()` so material sample disagreement independently forces `VERIFY` and hides the score at the UI boundary. The result card now explains sample disagreement and the technical panel reports it.

## Ground-truth verification output

### Dependency inventory and comparison

The final `pip list` reported the compatibility-sensitive versions:

```text
joblib       1.6.0
numpy        2.5.3
pandas       3.0.6
scikit-learn 1.9.1
torch        2.12.0+cu130
transformers 5.19.0
```

The explicit comparison against both requirements files reported:

```text
transformers installed=5.19.0 declared=5.19.0 MATCH
accelerate installed=1.15.0 declared=1.15.0 MATCH
safetensors installed=0.8.0 declared=0.8.0 MATCH
sentencepiece installed=0.2.2 declared=0.2.2 MATCH
numpy installed=2.5.3 declared=2.5.3 MATCH
pandas installed=3.0.6 declared=3.0.6 MATCH
scikit-learn installed=1.9.1 declared=1.9.1 MATCH
joblib installed=1.6.0 declared=1.6.0 MATCH
pyyaml installed=6.0.3 declared=6.0.3 MATCH
torch installed=2.12.0+cu130 declared=2.12.0+cu130 MATCH
dependency_mismatches= 0
```

### Syntax and tests

The literal wildcard form from the original checklist is not accepted by this PowerShell/Python invocation and produced `Errno 22: Invalid argument: 'src\\gate\\*.py'`. The same requested file scope was then expanded by PowerShell and passed:

```text
py_compile: PASS
```

Final full suite:

```text
........................................................................ [ 78%]
....................                                                     [100%]
92 passed in 4.70s
```

### Verifier smoke test

```text
PASS local: expected=VERIFIED, actual=VERIFIED | What is 15% of 240?
PASS local: expected=CONTRADICTED, actual=CONTRADICTED | What is 15% of 240?
PASS local: expected=UNSUPPORTED, actual=UNSUPPORTED | Who is Isaac Newton?
PASS live: expected=VERIFIED, actual=VERIFIED | What is the capital of Australia?
Verifier smoke result: 4/4 checks passed
```

### Real-model smoke test

The final bounded smoke test loaded `Qwen/Qwen3-0.6B` on CUDA with `NVIDIA GeForce GTX 1650` and reported:

```text
prompt=qa-useful-v5 fingerprint=763d5cfaf7b17341
diagnostic samples=1 max_new_tokens=48
consensus: 'The capital of Australia is Canberra.'
quality_ok=True useful=1/1 material_disagreement=False
evaluation=quality_and_expected_term_pass latency=2.3s -> PASS
1/1 passed.
```

### V5 artifact integration

```text
[PASS] Saved fingerprint matches live V5 config — saved='763d5cfaf7b17341' live='763d5cfaf7b17341'
[PASS] generation_config dict matches live config
[PASS] Training uses complete source rows and unique held-out questions — source=300, train=238, heldout=60, unique=298, duplicates_excluded=2
[PASS] Held-out acceptance metrics meet the recorded target and fingerprint — accuracy=0.9387755102040817, mean_trusted=49.0
[PASS] Artifact passes calibration check
[PASS] predict_proba returns finite probability in [0,1] — p=0.9085
V5 ARTIFACT INTEGRATION: 15/15 checks passed
All checks PASSED. V5 artifact is compatible with the production gate.
```

## Live sanity checks

All checks used the real local Qwen model and the real verifier. The Streamlit checks used `streamlit.testing.v1.AppTest` against the actual `app.py`; the command reported `APP_BOOT {'exceptions': 0, 'theme': 'light'}`.

1. `What is 15% of 240?`

   The live script returned five identical `15% of 240 is 36.` samples, `P(correct)=0.9666`, threshold `0.50`, `TRUST`, `reliability_reported=True`, verifier `VERIFIED`, `material_disagreement=False`, and `calibrated=True`. The Streamlit result also reported `TRUST`, `score_valid=True`, and `VERIFIED`.

2. `What is the capital of Australia?`

   A real five-sample run returned a Sydney/Canberra split, consensus Sydney, `P(correct)=0.2869`, `VERIFY`, `reliability_reported=False`, verifier `CONTRADICTED`, and `material_disagreement=True`. In the final Streamlit run, an Australia sample split produced `material_disagreement=True`, `VERIFY`, `score_valid=False`, and the UI displayed the sample-disagreement message and `N/A`; a second run also produced `material_disagreement=True` and `VERIFY` with verifier `CONTRADICTED`.

3. `who is rahul gandhi`

   The final real script run produced mixed biographical answers, including the captured BJP claim. Its gate result was:

   ```text
   P(correct)          : N/A (reliability not reported; raw diagnostic=0.7161)
   Decision            : VERIFY
   Reliability reported: False
   Verifier status     : UNSUPPORTED
   Verifier supported  : False
   Verifier reason     : Independent verification is not implemented for this question type yet; the gate must VERIFY.
   Material disagreement: False
   Calibrated          : True
   ```

   The post-fix Streamlit run also returned `VERIFY`, `score_valid=False`, `UNSUPPORTED`, and rendered both the `UNSUPPORTED QUESTION TYPE` label and `N/A`. The UI did not imply that the answer was reliable.

## Latency anomaly

The historical `smoke_v5.json` record contains a `1524.1s` Isaac Newton generation. A current five-sample reproduction with the same 48-token limit completed in `22.7s`; individual sample latencies were `4.622s`, `4.127s`, `3.301s`, `5.280s`, and `5.250s`. The anomaly was not reproducible, so no speculative timeout or hardware change was added. The app already displays per-run latency and surfaces GPU/inference failures.

## Remaining intentional limitations

- The verifier supports only percent-of arithmetic and capital lookups. Arbitrary biographical, political, authorship, and other claims remain `UNSUPPORTED -> VERIFY` by design.
- Model agreement is not independent evidence; a unanimous wrong answer can still occur.
- Wikidata is a reference source, not an infallible truth oracle. Network failure produces `UNAVAILABLE -> VERIFY`.
- The generated V5 classifier artifact is not committed because it is a machine-local binary; absence or fingerprint mismatch disables calibrated `TRUST` rather than silently using stale calibration.
