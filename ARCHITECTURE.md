# Architecture

Question
  -> Qwen3-0.6B x 5 sampled answers
  -> answer extraction + response-quality checks
  -> 15 uncertainty features + conservative agreement grouping
  -> calibration, fingerprint, and schema validation
  -> classifier probability
  -> narrow independent verifier
  -> TRUST only if calibrated + useful + agreed + above threshold + VERIFIED
     otherwise VERIFY

The verifier is intentionally narrow: it checks percentage-of arithmetic locally
and capital claims using Wikidata entity metadata, current P36 claims, labels,
and date/rank filtering. It returns VERIFIED, CONTRADICTED, UNSUPPORTED,
UNVERIFIED, or UNAVAILABLE; every result other than VERIFIED is a fail-closed
VERIFY condition.

The classifier score is a calibrated reliability estimate only when its artifact
matches the current model, prompt, generation settings, feature schema, and
verifier rule. Multiple samples from one model are not independent proof.
