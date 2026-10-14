# Local LLM Reliability Gate

## Current stage

The application runs a local Qwen3-0.6B model, samples five answers, extracts
uncertainty features, and applies a calibrated classifier when a compatible
artifact is available. A separate narrow verifier is also used before TRUST:

Question -> five sampled answers -> features -> calibration-checked classifier
         -> independent verifier -> TRUST only when every trust condition passes
                                      otherwise VERIFY

The independent verifier supports percentage-of arithmetic and capital questions
backed by current Wikidata P36 statements. It distinguishes VERIFIED,
CONTRADICTED, UNSUPPORTED, UNVERIFIED, and UNAVAILABLE. Unsupported, ambiguous,
stale, failed, or unavailable checks fail closed to VERIFY.

This is a reliability/abstention gate, not a universal truth detector.

## Verification

Run from the repository root using the project environment:

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe scripts\smoke_test_verifier.py

The regression suite is hermetic. The verifier smoke script reports deterministic
arithmetic separately from the live Wikidata Australia case and returns a nonzero
exit code if the live check is unavailable or fails.

Start the application with:

    .\.venv\Scripts\python.exe -m streamlit run app.py

Calibration metadata must match the current model, prompt, generation settings,
feature schema, and verifier rule. Missing or stale calibration may be diagnosed
by the application, but it cannot authorize TRUST.

The checked-in classifier is a V4 artifact while the live configuration is V5,
so it is intentionally detected as stale and remains fail-closed until a
legitimate compatible artifact is regenerated.
