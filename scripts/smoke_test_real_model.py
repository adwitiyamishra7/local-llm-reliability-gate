"""Real-model smoke test (needs Qwen3-0.6B locally). Run from the project root:

    python scripts/smoke_test_real_model.py

Uses the same prompt, sampling settings and response-quality check as the app by default. It
does not need a classifier file and reports NO reliability percentage. It never counts a refusal
as a success. The default is deliberately one question and one sample; use --all explicitly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gate.config import (  # noqa: E402
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    PROMPT_VERSION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    config_fingerprint,
    generation_config,
)
from src.gate.generation import generate_one  # noqa: E402
from src.gate.model import load_local_llm  # noqa: E402
from src.gate.reliability_gate import ReliabilityGate  # noqa: E402

# (question, must_be_useful, expected substring or None)
CASES = [
    ("Who is Narendra Modi?", True, "modi"),
    ("Who is Isaac Newton?", True, "newton"),
    ("What is the capital of Australia?", True, "canberra"),
    ("Who wrote Hamlet?", True, "shakespeare"),
    ("What is 15% of 240?", True, "36"),
    ("What is the name of my neighbour's cat?", False, None),  # abstention is the CORRECT outcome
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded real-model reliability diagnostic.")
    parser.add_argument("--question", default=CASES[2][0], help="One question to diagnose.")
    parser.add_argument(
        "--expected-contains",
        default=None,
        help="Optional expected substring for a custom question; without it, factual accuracy is not asserted.",
    )
    parser.add_argument("--all", action="store_true", help="Run all smoke cases; still uses --samples per case.")
    parser.add_argument("--samples", type=int, default=1, help="Samples per question (default: 1).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output", default="smoke_v5_diagnostic.json")
    args = parser.parse_args()
    if not 1 <= args.samples <= 5:
        parser.error("--samples must be between 1 and 5")
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS:
        parser.error(f"--max-new-tokens must be between 1 and {MAX_NEW_TOKENS}")
    return args


def diagnostic_samples(llm, question: str, sample_count: int, max_new_tokens: int):
    """Generate a deliberately bounded set while retaining live defaults."""
    return [
        generate_one(
            llm=llm,
            question=question,
            max_new_tokens=max_new_tokens,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        )
        for _ in range(sample_count)
    ]


def sample_diagnostic(result) -> dict[str, object]:
    return {
        "latency_s": round(result.latency_seconds, 3),
        "input_tokens": result.input_token_count,
        "generated_tokens": result.generated_token_count,
        "reached_token_limit": result.reached_token_limit,
        "stop_reason": result.stop_reason,
    }


_ABSTENTION_PATTERNS = (
    r"\bi do not know\b",
    r"\bi don't know\b",
    r"\bi cannot (?:tell|determine|identify|know)\b",
    r"\bi can't (?:tell|determine|identify|know)\b",
    r"\bi am not familiar\b",
    r"\bi'm not familiar\b",
    r"\bnot provided\b",
    r"\bnot specified\b",
    r"\bnot stated in the question\b",
    r"\bnot enough information\b",
    r"\bno information (?:is )?provided\b",
    r"\bcannot be determined\b",
    r"\bcan't be determined\b",
    r"\bunknown to me\b",
)


def is_abstention_answer(answer: str) -> bool:
    """Recognize common explicit limitations in smoke-test answers only."""
    text = str(answer or "").strip().casefold()
    return any(re.search(pattern, text) for pattern in _ABSTENTION_PATTERNS)


def evaluate_smoke_case(
    answer: str,
    quality_ok: bool,
    must_be_useful: bool,
    expected_needle: str | None,
) -> tuple[bool, str]:
    """Evaluate usefulness/expected term without treating an absent needle as failure."""
    if must_be_useful:
        if not quality_ok:
            return False, "quality_check_failed"
        if expected_needle:
            if expected_needle.casefold() not in answer.casefold():
                return False, "expected_term_missing"
            return True, "quality_and_expected_term_pass"
        return True, "quality_pass_accuracy_not_asserted"

    if is_abstention_answer(answer):
        return True, "abstention_detected"
    return False, "expected_abstention_not_detected"


def main() -> int:
    args = parse_args()
    print(f"prompt={PROMPT_VERSION} fingerprint={config_fingerprint()}")
    print(f"diagnostic samples={args.samples} max_new_tokens={args.max_new_tokens}")
    llm = load_local_llm()
    rows, failures = [], 0
    if args.all:
        cases = CASES
    else:
        # The default question has a known smoke-test expectation. A custom
        # question without --expected-contains checks usefulness only, not truth.
        needle = args.expected_contains
        if needle is None and args.question == CASES[2][0]:
            needle = CASES[2][2]
        cases = [(args.question, True, needle)]

    for question, must_be_useful, needle in cases:
        t0 = perf_counter()
        results = diagnostic_samples(llm, question, args.samples, args.max_new_tokens)
        latency = perf_counter() - t0
        answers = [r.answer.strip() for r in results]
        answer, info = ReliabilityGate._select_consensus(answers, question)
        ok, evaluation = evaluate_smoke_case(
            answer=answer,
            quality_ok=bool(info["quality_ok"]),
            must_be_useful=must_be_useful,
            expected_needle=needle,
        )
        failures += 0 if ok else 1
        rows.append(dict(
            question=question,
            answers=answers,
            consensus=answer,
            quality_ok=info["quality_ok"],
            reason=info["quality_reason"],
            useful_samples=info["useful_samples"],
            material_disagreement=info.get("material_disagreement", False),
            expected_contains=needle,
            accuracy_asserted=bool(needle) if must_be_useful else None,
            evaluation=evaluation,
            unfinished=[not r.finished for r in results],
            sample_diagnostics=[sample_diagnostic(r) for r in results],
            latency_s=round(latency, 1),
            ok=ok,
        ))
        accuracy_note = (
            f" expected={needle!r}" if needle else " accuracy=NOT ASSERTED"
        )
        print(
            f"\nQ: {question}\n  consensus: {answer!r}\n  quality_ok={info['quality_ok']} "
            f"useful={info['useful_samples']}/{len(answers)} "
            f"material_disagreement={info.get('material_disagreement', False)}{accuracy_note} "
            f"evaluation={evaluation} latency={latency:.1f}s -> {'PASS' if ok else 'FAIL'}"
        )
        for index, (a, result) in enumerate(zip(answers, results), start=1):
            print(f"    - sample {index}: {a!r} | {result.latency_seconds:.3f}s | "
                  f"input={result.input_token_count} generated={result.generated_token_count} "
                  f"stop={result.stop_reason}")
    out = ROOT / "artifacts" / "results" / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": generation_config(),
        "diagnostic": {
            "questions": len(cases),
            "samples_per_question": args.samples,
            "max_new_tokens": args.max_new_tokens,
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n{len(cases) - failures}/{len(cases)} passed. Details: {out}")
    print("Note: a custom question without --expected-contains checks usefulness only, not factual accuracy. "
          "PASS is not a general accuracy guarantee.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
