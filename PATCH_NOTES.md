# Local LLM Reliability Gate — conflict-grouping patch (V5.1)

## What this patch fixes

The uploaded V5 snapshot's `answers_are_equivalent()` grouped answers using substring containment and, for `Who is X?` prompts, the fact that both answers began with the same subject. Those heuristics could merge obviously conflicting statements such as:

- `Canberra` vs `Canberra is not the capital; Sydney is the capital.`
- `Isaac Newton was born in 1900.` vs `Isaac Newton was born in 1643.`

The patch adds a conservative pre-check for explicit negation/contrast markers and conflicting answer numbers (excluding numbers already present in the question) before equivalence grouping. If a likely conflict is detected, the two answers remain in different groups, so `material_disagreement` forces `VERIFY`.

The active config's `answer_grouping_rule` identifier is bumped. This deliberately changes the configuration fingerprint, so an artifact trained under the old grouping semantics should not be considered compatible.

## Files in this patch

- `src/gate/agreement.py`
- `src/gate/config.py`
- `tests/test_agreement.py`
- `tests/test_gate_v5.py`

## Validation performed

- `python -m py_compile src/gate/agreement.py src/gate/config.py tests/test_agreement.py tests/test_gate_v5.py`
- `python -m pytest tests -q`: **55 passed** in the bundled test environment.

These are unit tests, not a real-Qwen run. This patch does not solve the fundamental case where every generated sample confidently gives the same wrong answer (for example, all samples say Sydney). Detecting a unanimous wrong answer requires an independent trusted source/verifier or a validated probabilistic reliability estimate; self-consistency alone cannot prove factual truth.

## Apply on Windows

From `E:\LLM GATE\local-llm-reliability-gate`:

```powershell
$backup = "backups\pre_grouping_v51_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $backup | Out-Null
Copy-Item src\gate\agreement.py, src\gate\config.py, tests\test_agreement.py, tests\test_gate_v5.py $backup
Expand-Archive -LiteralPath .\llm_gate_conflict_grouping_patch_v51.zip -DestinationPath . -Force
.\.venv\Scripts\python.exe -m py_compile src\gate\agreement.py src\gate\config.py tests\test_agreement.py tests\test_gate_v5.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

The original V4 artifacts are not included in this patch and are not overwritten. Because the fingerprint changes, any V5 calibration created before this patch is stale and should be regenerated before scores are reported as calibrated.
