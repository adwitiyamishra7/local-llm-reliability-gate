# Run from E:\LLM GATE\local-llm-reliability-gate  (extract the ZIP contents into this folder first, overwriting app.py)
# Restore: copy backups\pre_ui_fix\app.py back over app.py
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force backups\pre_ui_fix | Out-Null
# NOTE: run this backup line BEFORE extracting the ZIP over app.py:
#   Copy-Item app.py backups\pre_ui_fix\app.py
python -m pip install pytest
python -m pytest tests\test_ui_regression.py -q
streamlit run app.py --server.fileWatcherType none
