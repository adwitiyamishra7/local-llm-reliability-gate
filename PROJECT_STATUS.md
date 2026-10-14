# Project Status

## Implemented

- Local Qwen3-0.6B generation with token-level confidence features.
- Five-sample answer grouping with response-quality and disagreement guards.
- Calibration-checked classifier loading with feature and configuration integrity checks.
- Fail-closed independent verification for percentage arithmetic and current Wikidata capitals.
- Streamlit application using the same production gate and verifier path.
- Hermetic regression tests plus a separate live Wikidata smoke test.

## Decision policy

TRUST requires compatible calibration, useful output, no material sampled-answer
disagreement, a classifier score at or above the artifact threshold, and
independent verification status VERIFIED.

CONTRADICTED, UNSUPPORTED, UNVERIFIED, UNAVAILABLE, missing calibration, stale
fingerprints, malformed features, and integration failures all produce VERIFY.
Agreement among samples is not proof of factual correctness.

## Current artifact status

The checked-in classifier is a V4 artifact trained with an older prompt and
generation configuration. The live configuration is V5 and includes the current
verifier rule, so the artifact is detected as stale and cannot authorize TRUST.
No labels or calibration results are fabricated; regenerate it only through the
existing benchmark/training pipeline with compatible legitimate data.

## Verification commands

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts\smoke_test_verifier.py
    .\.venv\Scripts\python.exe -m streamlit run app.py
