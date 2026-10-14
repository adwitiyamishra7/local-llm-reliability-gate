# V5.2 Final Cleanup Report

Date: 2026-10-14

The cleanup started from a clean worktree at `6f9b35f` (`v5.2-ui-fixes-final`). The scope remained limited to the requested UI verification and factual-evidence presentation cleanup.

## (a) Reliability circle on disagreement

The circle was already correct. A real Streamlit AppTest submitted `What is the capital of Australia?` and produced:

```text
RESULT_FIELDS {'decision': 'VERIFY', 'probability': 0.2868968916653168, 'score_valid': False, 'material_disagreement': True, 'fact_verification_status': 'CONTRADICTED'}
EXCEPTIONS []
CIRCLE_MARKERS {'not_reported': True, 'n_a': True, 'numeric_28_7': False}
```

The actual rendered circle HTML contained:

```html
<text x="140" y="154" text-anchor="middle" font-size="50" font-weight="800" fill="#0f172a"
style="font-family:inherit">N/A</text>
<text x="140" y="180" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="#64748b"
style="font-family:inherit">NOT REPORTED</text>
<text x="140" y="206" text-anchor="middle" font-size="14" font-weight="800" letter-spacing="2" fill="#b45309"
style="font-family:inherit">VERIFY</text>
<div class="meter-fill verify" style="--w:0.00%"></div>
```

No `app.py` change was needed. The internal raw diagnostic probability does not leak into the rendered Reliability Estimate circle when `material_disagreement=True`.

## (b) Capital-evidence alias display

Real Wikidata smoke output before the patch:

```text
Wikidata lists the current capital: Canberra. Alternate name(s)/aliases: Canberra, Australia, Canberra, Australian Capital Territory.
```

After the patch, the same live Australia check produced:

```text
Wikidata lists the current capital: Canberra. Alternate name(s)/aliases: Canberra, Australia, Canberra, Australian Capital Territory.
```

The live record currently supplies the two compound aliases `Canberra, Australia` and `Canberra, Australian Capital Territory`; it does not supply a separate exact `Canberra` alias, so the live sentence is unchanged. The display assembly now nevertheless deduplicates aliases case-insensitively and removes any alias whose normalized value equals the canonical capital label.

The hermetic display regression injects `Canberra`, `CANBERRA`, `Canberra City`, and `canberra city`. Its rendered evidence is exactly:

```text
Wikidata lists the current capital: Canberra. Alternate name(s)/aliases: Canberra City.
```

Canonical and all alias variants remain in `expected_values`, so matching behavior is unchanged.

## (c) Regression checks

The full hermetic suite passed:

```text
98 passed in 6.45s
```

Real clean TRUST path:

```text
Question: What is 15% of 240?
P(correct)          : 0.9666
Decision            : TRUST
Reliability reported: True
Verifier status     : VERIFIED
Material disagreement: False
```

Real Canberra/Sydney disagreement path:

```text
1. The capital of Australia is Canberra.
2. The capital of Australia is Canberra.
3. The capital of Australia is Canberra.
4. The capital of Australia is Canberra.
5. The capital of Australia is Sydney.
Decision            : VERIFY
Reliability reported: True
Verifier status     : VERIFIED
Material disagreement: True
```

## (d) Decision-policy boundary

`src/gate/reliability_gate.py` was not modified. Its decision and threshold logic, probability computation, calibration, and policy behavior were untouched.

