# V5.2 independent-verification patch

## Scope

This patch adds a small, conservative independent-verification layer. It is intentionally NOT a general fact checker.

- Percentage-of arithmetic such as `What is 15% of 240?` is checked locally with deterministic arithmetic.
- `What is the capital of X?` is checked against the capital (`P36`) statements exposed by Wikidata's Action API.
- Other question types return `UNSUPPORTED` and the gate must return `VERIFY`.
- Contradicted, ambiguous, unverified, unsupported, or unavailable verifier results cannot receive `TRUST` or a reported reliability score.
- The app displays verifier status, a short reason/evidence, and a source link where available.
- The independent-verification policy is part of the config fingerprint, so previous V4/V5 artifacts are not assumed compatible.

## Network and privacy

The capital verifier makes public requests to Wikidata. It sends only the extracted subject name (for example, `Australia`) rather than the full user question or model answer. It does not use an API key or add a Python dependency. Configure a descriptive `WIKIDATA_USER_AGENT` including your own contact address or project URL for repeated use. If the service cannot be reached, status becomes `UNAVAILABLE` and the gate fails closed.

Wikidata is a reference source, not an infallible oracle. The parser supports a narrow set of clear capital statements. Complex wording, ambiguous entities or unsupported relations return a non-verified result rather than attempting broad semantic inference.

## Known limitation

This patch does not make Narendra Modi, Isaac Newton, authorship, or arbitrary natural-language claims independently verifiable. Those questions will show `UNSUPPORTED` and remain `VERIFY` until suitable verifiers are added. This is deliberate.

## Validation performed here

- Python compile checks passed.
- 73 unit tests passed in the patch workspace.
- Wikidata HTTP connectivity was NOT verified from the build sandbox because DNS/network access was unavailable. Run `scripts/smoke_test_verifier.py` on the Windows machine to check live connectivity.
- No real Qwen generation and no V5 training/finalization was run here.
