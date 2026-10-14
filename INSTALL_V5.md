# V5 install (Windows PowerShell) — run from E:\LLM GATE\local-llm-reliability-gate

1. Backup (restore = copy these back):
   New-Item -ItemType Directory -Force backups\pre_v5_gen | Out-Null
   Copy-Item app.py, src\gate\config.py, src\gate\generation.py, src\gate\reliability_gate.py, src\gate\response_quality.py backups\pre_v5_gen\
2. Extract the ZIP over the project (paths are relative: app.py, src\gate\*.py, tests\*, scripts\smoke_test_real_model.py).
   V4 artifacts (artifacts\classifier\reliability_gate_final_v4.pkl etc.) are NOT touched.
3. Test:  python -m pip install pytest ; python -m pytest tests -q
4. Real model:  python scripts\smoke_test_real_model.py
5. App:  streamlit run app.py --server.fileWatcherType none
Until a V5 gate file exists, every answer is VERIFY and no percentage is shown (V4 calibration is stale).
