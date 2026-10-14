# Install V5.2 independent verification (Windows PowerShell)

Run from the project root:

`E:\LLM GATE\local-llm-reliability-gate`

## 1. Make a timestamped backup

```powershell
$backup = "backups\pre_v52_verification_$(Get-Date -Format yyyyMMdd_HHmmss)"
$files = @(
  "app.py",
  "src\gate\agreement.py",
  "src\gate\config.py",
  "src\gate\reliability_gate.py",
  "scripts\smoke_test_real_model.py",
  "tests\test_agreement.py",
  "tests\test_gate_v5.py",
  "tests\test_smoke_test_regression.py",
  "tests\test_ui_regression.py"
)
foreach ($file in $files) {
  if (Test-Path $file -PathType Leaf) {
    $dest = Join-Path $backup $file
    New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
    Copy-Item $file $dest -Force
  }
}
```

## 2. Extract the patch ZIP over the project root

Place `llm_gate_independent_verification_v52.zip` in this folder, then run:

```powershell
Expand-Archive -LiteralPath .\llm_gate_independent_verification_v52.zip -DestinationPath . -Force
```

The patch does not include model weights, `.venv`, secrets, or V4 classifier artifacts. It does not overwrite V4 model/data artifacts.

## 3. Run compilation and automated tests

```powershell
.\.venv\Scripts\python.exe -m py_compile app.py src\gate\agreement.py src\gate\config.py src\gate\factual_verifier.py src\gate\reliability_gate.py scripts\smoke_test_real_model.py scripts\smoke_test_verifier.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 4. Test the verifier without loading Qwen

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_verifier.py
```

The math tests run locally. The capital check requires Internet access to Wikidata. For repeated use, set a descriptive User-Agent with your own contact address/URL to comply with Wikimedia API usage guidance, for example (replace the placeholder):

```powershell
$env:WIKIDATA_USER_AGENT = "LocalLLMReliabilityGate/0.6 (contact: YOUR_PUBLIC_EMAIL_OR_PROJECT_URL)"
.\.venv\Scripts\python.exe scripts\smoke_test_verifier.py
```

If it reports `UNAVAILABLE`, the gate must stay on `VERIFY`; do not disable fail-closed behavior to get a green result.

## 5. Launch the app

```powershell
streamlit run app.py --server.fileWatcherType none
```

## Important

The old classifier calibration is not valid for the changed grouping/verifier fingerprint. Until a compatible V5 benchmark and finalizer produce a matching artifact, the app should show `VERIFY` and suppress calibrated reliability percentages. The verifier only independently checks percentage-of arithmetic and capital-of questions; other question types remain `VERIFY`.
