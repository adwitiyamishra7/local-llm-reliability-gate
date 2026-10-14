# Local LLM Reliability Gate

This project is a local reliability and abstention gate for a small language model. It runs Qwen3-0.6B locally, samples five answers, extracts 15 uncertainty features, and applies a calibration-checked ExtraTrees classifier. A separate, deliberately narrow verifier must also support and verify the claim before the system can recommend `TRUST`.

The design is intentionally conservative:

```text
question -> five local samples -> quality/agreement checks -> 15 features
         -> compatible calibrated classifier -> independent verifier
         -> TRUST only when every condition passes; otherwise VERIFY
```

This is not a universal truth detector. Agreement and classifier confidence are signals about reliability, not proof that a claim is true.

## Current behavior

The live configuration is `qa-useful-v5` with five samples, a 48-token generation limit, temperature `0.7`, top-p `0.8`, and top-k `20`. The local V5 artifact is `artifacts/classifier/reliability_gate_final_v5.pkl`; it is intentionally ignored by Git because generated model artifacts are machine-local. When present, the application validates its schema, feature names, generation configuration, fingerprint, and threshold before using it.

The independent verifier supports only:

- percentage-of arithmetic, checked deterministically;
- clear capital-of claims, checked against current Wikidata capital statements.

Other question types return `UNSUPPORTED`. `UNSUPPORTED`, `CONTRADICTED`, `UNVERIFIED`, `UNAVAILABLE`, stale calibration, poor response quality, and material sample disagreement all fail closed to `VERIFY`. For those cases the UI labels the reliability estimate `N/A`; any raw classifier value is diagnostic only and is not presented as a reported probability.

For example, `who is rahul gandhi` is outside the verifier's scope. Even if the model produces a confident biographical error such as assigning Rahul Gandhi to the BJP, the result remains `VERIFY` with `UNSUPPORTED` verification and no reported reliability percentage. That behavior is covered by a hermetic regression test.

## Run on Windows

The validated environment uses Python 3.13.5, PyTorch 2.12.0+cu130, Transformers 5.19.0, scikit-learn 1.9.1, and joblib 1.6.0. Install the GPU-pinned environment when setting up a fresh checkout:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu-cu130.txt
```

Start the Streamlit application from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The root `.streamlit/config.toml` selects the light theme. The first reliability check loads the local model; subsequent checks reuse it in the process. No cloud model API is used. A descriptive `WIKIDATA_USER_AGENT` may be set for repeated live capital checks.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\smoke_test_verifier.py
.\.venv\Scripts\python.exe scripts\test_v5_artifact.py
.\.venv\Scripts\python.exe scripts\smoke_test_real_model.py
```

For one real five-sample gate run, including verifier status and whether reliability was reported:

```powershell
.\.venv\Scripts\python.exe scripts\test_live_gate.py --question "What is 15% of 240?"
.\.venv\Scripts\python.exe scripts\test_live_gate.py --question "who is rahul gandhi"
```

The V5 artifact is required for calibrated scoring. If it is absent or incompatible, the application remains fail-closed and reports that calibration is unavailable; it does not silently use a stale artifact.

## Scope and limitations

The model can produce a fluent, specific, wrong answer. Sampling disagreement can catch some uncertainty, but unanimous model output is not independent evidence. The verifier therefore remains narrow by design. Extending it to a new claim type requires a real reference-backed verifier, tests, and a deliberate configuration/fingerprint update; heuristic guesses are not verification.

The application displays generation latency and surfaces model/GPU failures. A rare long generation should be investigated as an environment or driver issue rather than hidden by an unbounded silent timeout.
