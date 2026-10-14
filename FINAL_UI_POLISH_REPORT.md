# UI/UX Polish Report

## Changes

- The pre-load calibration notice now uses an informational style and says that the model loads automatically on the first check. The warning style remains for a loaded but uncalibrated gate.
- TRUST and VERIFY decision badges are flat, outlined, pill-shaped, and noninteractive. They have no shadow or pointer affordance.
- The existing stage hooks are now on the live execution path. The UI reports model loading, sampling, feature extraction, classification, verification, and finalization as those backend boundaries execute. Each stage includes an indeterminate live indicator and elapsed time.
- The question control is now a single-line `st.text_input` with `on_change`, so Enter sets the same submit path as the Check Reliability button. The existing button and clear controls remain available.

## Evidence

- `python -m pytest -q -p no:cacheprovider`: **98 passed in 10.47s**.
- `python -m py_compile app.py`: passed.
- `git diff --check`: passed.
- Helper interaction checks passed for button/Enter submission, pre-load informational state, post-load warning state, and calibrated empty state.
- The staged loading path is wired through `load_and_run()` and `run_with_stage_hooks()`; no timer or fabricated percentage is used.

## Scope and regressions

Only `app.py` and this report were changed. No files under `src/gate/` were modified, so decision logic, thresholds, verification scope, and model behavior remain unchanged. The existing lazy load, calibration validation, alias deduplication, TRUST path, and material-disagreement VERIFY behavior remain covered by the passing suite.

Factual accuracy for unsupported question types such as geography and biography is unchanged and remains intentionally out of scope. The model continues to fail closed where the existing gate requires verification.

The Enter behavior uses Streamlit's supported `st.text_input` event path, which trades the former multiline textarea for a search-style single-line question field as permitted by the original request's robust implementation option.
