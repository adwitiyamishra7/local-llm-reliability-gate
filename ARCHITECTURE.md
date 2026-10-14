# Architecture

## End-to-end path

```text
User question
    |
    v
Qwen3-0.6B, five sampled answers
    |
    +--> response-quality checks and cleaned answer text
    +--> conservative equivalence grouping
    |
    v
15 uncertainty features
    |
    v
V5 ExtraTrees classifier and calibration/fingerprint checks
    |
    v
P(correct) as a model signal
    |
    v
Narrow independent verifier
    |
    v
TRUST only when all gates pass; otherwise VERIFY
```

## Generation and features

`src/gate/config.py` is the single source of truth for the model name, prompt, sampling settings, feature order, paths, and configuration fingerprint. `src/gate/generation.py` produces one bounded answer with token-level log-probability diagnostics. `src/gate/reliability_gate.py` samples exactly five answers and computes:

- mean and minimum log-probability summaries;
- confidence range;
- answer agreement, consensus margin, and answer-key entropy;
- answer length statistics;
- truncation rate.

Response-quality checks prevent refusals, echoed questions, empty output, and obvious truncation from being treated as useful answers. The agreement layer blocks conservative grouping when answers contain explicit contradiction markers or conflicting factual numbers. Multiple samples from the same model are not independent proof.

## Calibration and decision policy

The V5 artifact stores the ExtraTrees model, feature schema, trust threshold, generation configuration, fingerprint, and benchmark provenance. The live gate rejects missing or stale calibration, feature mismatches, malformed probabilities, and model/configuration mismatches.

The decision is equivalent to:

```text
TRUST iff
    artifact is compatible and calibrated
    AND the consensus is useful
    AND non-answer output is not the majority
    AND useful samples have no material disagreement
    AND P(correct) >= the artifact threshold
    AND independent_verification.status == VERIFIED
otherwise VERIFY
```

`probability_correct` may still exist internally as a diagnostic value after a failed gate, but `reliability_reported` is false and the UI renders the estimate as `N/A` unless calibration, quality, and independent verification all pass.

## Independent verifier

`src/gate/factual_verifier.py` intentionally supports only two narrow families:

1. percentage-of arithmetic, checked locally with deterministic calculation;
2. capital-of questions, checked against Wikidata search/entity metadata and current P36 statements.

Unsupported questions do not make network requests and return `UNSUPPORTED`. A contradicted, ambiguous, unavailable, or otherwise incomplete result cannot authorize `TRUST`. Wikidata is a reference source, not an infallible oracle; parsing is deliberately limited to clear capital claims.

## Application boundary

The Streamlit app imports the gate lazily so the page can render before model loading. `build_view()` validates the result again at the UI boundary and fails closed if trust-critical fields are missing or inconsistent. The result card shows the final decision, verifier status/reason/evidence, and a reliability gauge only when the score is reportable. The root `.streamlit/config.toml` owns the page theme.

## Operational behavior

The model is local and prefers CUDA when available. Generation latency is recorded per run. GPU out-of-memory and other inference failures are surfaced to the user; they are not converted into a false `TRUST` or swallowed as a successful result.
