"""Real-model smoke test (needs Qwen3-0.6B locally). Run from the project root:

    python scripts/smoke_test_real_model.py

Uses the SAME sample_answers() / prompt / quality check as the app. It does not need a
classifier file and it reports NO reliability percentage. It never counts a refusal as a success.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gate.config import PROMPT_VERSION, config_fingerprint, generation_config  # noqa: E402
from src.gate.model import load_local_llm  # noqa: E402
from src.gate.reliability_gate import ReliabilityGate, sample_answers  # noqa: E402

# (question, must_be_useful, expected substring or None)
CASES = [
    ("Who is Narendra Modi?", True, "modi"),
    ("Who is Isaac Newton?", True, "newton"),
    ("What is the capital of Australia?", True, "canberra"),
    ("Who wrote Hamlet?", True, "shakespeare"),
    ("What is 15% of 240?", True, "36"),
    ("What is the name of my neighbour's cat?", False, None),  # abstention is the CORRECT outcome
]


def main() -> int:
    print(f"prompt={PROMPT_VERSION} fingerprint={config_fingerprint()}")
    llm = load_local_llm()
    rows, failures = [], 0
    for question, must_be_useful, needle in CASES:
        t0 = perf_counter()
        results = sample_answers(llm, question)
        latency = perf_counter() - t0
        answers = [r.answer.strip() for r in results]
        answer, info = ReliabilityGate._select_consensus(answers, question)
        contains = (needle in answer.lower()) if needle else None
        if must_be_useful:
            ok = info["quality_ok"] and bool(contains)
        else:
            ok = True  # informational: ideally quality_ok is False (honest abstention)
        failures += 0 if ok else 1
        rows.append(dict(question=question, answers=answers, consensus=answer, quality_ok=info["quality_ok"],
                         reason=info["quality_reason"], useful_samples=info["useful_samples"],
                         unfinished=[not r.finished for r in results], latency_s=round(latency, 1), ok=ok))
        print(f"\nQ: {question}\n  consensus: {answer!r}\n  quality_ok={info['quality_ok']} "
              f"useful={info['useful_samples']}/{len(answers)} latency={latency:.1f}s -> {'PASS' if ok else 'FAIL'}")
        for a in answers:
            print(f"    - {a!r}")
    out = ROOT / "artifacts" / "results" / "smoke_v5.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": generation_config(), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed. Details: {out}")
    print("Note: PASS = useful consensus containing the expected term. It is not a general accuracy measure.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
