# Changelog

## V3 — initial reliability gate

- Established the local Qwen sampling pipeline, 15 uncertainty features, calibrated ExtraTrees scoring, and the `TRUST`/`VERIFY` abstention policy.
- Defined the project as a reliability estimate rather than a general truth detector.

## V4 — generation and calibration revision

- Versioned the prompt and generation settings and tied calibration to a configuration fingerprint.
- Added response-quality safeguards and kept the operating threshold selection separate from held-out evaluation.

## V5 — useful-answer consensus and reference labels

- Moved to the `qa-useful-v5` prompt and current sampling settings, with no retry/resampling policy.
- Added useful-answer consensus selection and reference-anchored benchmark labels; the V5 artifact records provenance and rejects configuration drift.

## V5.1 — conflict-guarded grouping

- Prevented substring or shared-subject heuristics from merging visibly contradictory answers, including conflicting numbers and explicit negation/contrast.
- Material disagreement now forces `VERIFY`, even when one candidate has a plurality.

## V5.2 — independent verification

- Added narrow deterministic percentage-math verification and Wikidata-backed capital verification.
- Required `VERIFIED` independent evidence for `TRUST`; unsupported, contradicted, unverified, unavailable, stale, and malformed cases fail closed.
- Added verifier status/evidence to the Streamlit result and regression coverage for unsupported biographical claims.

## Finalization pass

- Reconciled the documentation with the compatible local V5 artifact, moved the Streamlit theme to the project root, removed inert historical copies and dead parallel configuration, and retained one portable review-bundle script.
- Added a real-model diagnostic argument for reproducing one question at a time and a hermetic Rahul Gandhi regression fixture.
