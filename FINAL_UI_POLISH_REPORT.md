# UI/UX Polish Report

## Implemented

- The pre-load calibration notice is informational and explains that the local model loads on the first submitted reliability check. After loading, a valid calibration clears the notice; an invalid calibration shows its detected reason.
- TRUST and VERIFY remain flat, outlined status badges. Both expose `role="status"` and a final-decision label; the VERIFY icon now indicates an alert rather than an action. Their icons also use colors visible on the badge backgrounds.
- Loading updates now follow the real execution boundaries: first-load initialization, sampling, feature extraction, classifier inference, independent verification, and finalization. A spinner, indeterminate bar, and “Still working” message remain active during a long stage. The UI does not display a stale elapsed counter or a generic stage after inference has completed.
- The question and Check Reliability control now share a native Streamlit form. Enter submits the form; the explicit button uses that same submission path. The Clear control and example buttons remain available. Form submission avoids triggering inference merely because the text field loses focus.
- Streamlit documents Enter in `st.text_input` as submitting its containing form: [Forms: using Enter to submit](https://docs.streamlit.io/develop/concepts/architecture/forms).
- Gate hook cleanup now uses `finally` for every installed temporary patch, including when verification raises. Inference errors are caught by the UI, and the loaded/offline and calibration displays continue to reflect the actual gate state.

## Verification evidence

- Focused UI tests: `\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_ui_regression.py tests/test_ui_app.py` — **15 passed in 1.59s**.
- Full suite: `\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` — **102 passed in 7.86s**.
- Syntax: `\.venv\Scripts\python.exe -m py_compile app.py tests\test_ui_regression.py tests\test_ui_app.py` — passed.
- `git diff --check` — passed.
- Streamlit AppTest ran the actual app UI with a fake gate. It confirmed the initial informational notice and no initial model construction; an empty submit showed validation and still did not construct the gate. A valid form submission constructed one fake gate, produced one result, updated the banner to the supplied fingerprint-mismatch warning, and emitted the six expected stage labels in order. A plain rerun did not execute inference again; Clear reset the input/result; a later explicit submit reused the same gate. The test also confirms temporary stage hooks are restored after the run.
- Existing regression tests for a valid TRUST result and material-disagreement VERIFY passed in the full suite.

## Limits and scope

No browser automation package or browser executable is available in this environment. AppTest exercised the native form submit control, but did not send a literal keyboard Enter event in a real browser. The AppTest gate is a deterministic fake; Qwen/GPU inference was not run. Streamlit’s native form submission is the Enter implementation, but direct browser-key and real-model timing evidence remains unavailable here.

No files under `src/gate/` were changed. Gate decisions, classifier thresholds, calibration checks, verifier scope, model prompts, and model behavior are unchanged. Accuracy on unsupported geography and biography questions remains unchanged by design; the existing fail-closed VERIFY behavior remains in force.

The starting commit/tag `e9ca699` / `v5.3-ui-polish` is preserved. This follow-up does not move or recreate that tag.
