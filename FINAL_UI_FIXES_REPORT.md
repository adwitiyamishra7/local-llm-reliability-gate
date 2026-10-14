# V5.2 Targeted UI Follow-up Report

Date: 2026-10-14

Existing history was confirmed before this pass: `e2a8d9b` (`v5.2-final`), `f21fa9e`, and `097772b` (`pre-finalization-baseline`). Four pre-existing user worktree changes were preserved.

## (a) Mongolia disagreement

The disagreement was genuine, not a grouping artifact. A real Qwen run printed these raw samples before grouping:

```text
1: The capital of Mongolia is Nangaralt.
2: The capital of Mongolia is Ulaanbaatar.
3: The capital of Mongolia is Ulaanbaatar.
4: The capital of Mongolia is Kyungju.
5: The capital of Mongolia is Ulaanbaatar.
```

The actual grouping diagnostics were:

```text
sample 2 vs group 1: key_equal=False containment=False conflict_guard=False equivalent=False
sample 3 vs group 1: key_equal=False containment=False conflict_guard=False equivalent=False
sample 3 vs group 2: key_equal=True containment=True conflict_guard=False equivalent=True
sample 4 vs group 1: key_equal=False containment=False conflict_guard=False equivalent=False
sample 4 vs group 2: key_equal=False containment=False conflict_guard=False equivalent=False
sample 5 vs group 1: key_equal=False containment=False conflict_guard=False equivalent=False
sample 5 vs group 2: key_equal=True containment=True conflict_guard=False equivalent=True
GROUPS:
  group 1: samples [1]
  group 2: samples [2, 3, 5]
  group 3: samples [4]
```

Normalized keys were `nangaralt`, `ulaanbaatar`, and `kyungju`. The real gate result was `decision=VERIFY`, `material_disagreement=True`, `fact_verification_status=VERIFIED`, and `reliability_reported=True`. The grouping logic was not changed. A permanent hermetic test now covers this genuine disagreement in `tests/test_gate_v5.py`.

The existing Canberra/Sydney hermetic conflict check still passes:

```text
2 passed, 45 deselected in 3.14s
```

The post-fix real five-sample Canberra/Sydney run also remained fail-closed:

```text
samples: Sydney, Canberra, Canberra, Canberra, Sydney
P(correct): 0.2875
Decision: VERIFY
Verifier status: VERIFIED
Material disagreement: True
```

## (b) Issue A — stale calibration warning

Before the fix, the real Streamlit AppTest showed:

```text
BEFORE {'warning': True, 'warning_count': 1, 'exceptions': 0}
AFTER {'warning': True, 'warning_count': 1, 'decision': 'TRUST', 'fact_verification_status': 'VERIFIED', 'exceptions': 0}
```

So the initial lazy-load warning remained after the gate had loaded and produced a real `TRUST`/`VERIFIED` result.

The fix keeps lazy loading unchanged and gives the warning a placeholder that is updated immediately after `ensure_gate()` validates the artifact. After the fix, the real Streamlit AppTest showed:

```text
POST_FIX_BEFORE_CLICK {'warning': True, 'exceptions': 0}
POST_FIX_AFTER_MATH {'warning': False, 'decision': 'TRUST', 'fact_verification_status': 'VERIFIED', 'exceptions': 0}
```

The page still renders the initial warning instantly; it now clears it as soon as the loaded gate confirms calibration.

## (c) Issue B — capital evidence display

Before the display-only fix, the real verifier smoke output said:

```text
evidence: Wikidata lists the current capital value(s): Canberra, Canberra, Australia, Canberra, Australian Capital Territory.
```

After the fix, the same real Australia/Wikidata smoke output said:

```text
evidence: Wikidata lists the current capital: Canberra. Alternate name(s)/aliases: Canberra, Australia, Canberra, Australian Capital Territory.
```

The rendered Streamlit HTML for a real Australia result contained:

```html
<div>The answer asserts a capital value (&#x27;Sydney&#x27;) that conflicts with Wikidata for Australia. Claimed value: Sydney. Wikidata lists the current capital: Canberra. Alternate name(s)/aliases: Canberra, Australia, Canberra, Australian Capital Territory.</div>
```

Matching still uses the canonical label and aliases together; only the evidence wording and label/alias representation changed. A hermetic verifier test covers the separated presentation.

## (d) Regression checks

The clean live TRUST path remained unchanged:

```text
Question: What is 15% of 240?
P(correct): 0.9666
Decision: TRUST
Reliability reported: True
Verifier status: VERIFIED
Material disagreement: False
```

Final verification:

```text
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 3.77s
py_compile: PASS
Verifier smoke result: 4/4 checks passed
```

No classifier, threshold, retraining, recalibration, or decision-policy change was made in this targeted pass. The pre-existing strict fingerprint and benchmark-label edits were preserved.

## (e) Deployment boundary

The local browser demo is deployment-ready. Public internet deployment is explicitly out of scope for this pass and separately requires hosting, inference startup, and access-control work.
