from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.generation import generate_one
from src.gate.model import load_local_llm


def main() -> None:
    question = "What is the capital of Australia?"

    print("=" * 60)
    print("LOCAL LLM RELIABILITY GATE - STAGE 1 SMOKE TEST")
    print("=" * 60)

    llm = load_local_llm()

    print("\nQuestion:")
    print(question)

    result = generate_one(
        llm,
        question=question,
        max_new_tokens=32,
        do_sample=False,
    )

    print("\nAnswer:")
    print(result.answer)

    print(f"\nLatency: {result.latency_seconds:.3f} seconds")

    print("\nToken-level scores:")
    for token, logprob in zip(
        result.generated_tokens,
        result.token_logprobs,
    ):
        print(
            f"{token!r:20} "
            f"logprob={logprob:8.4f} "
            f"prob={math.exp(logprob):8.2%}"
        )

    print("\nSummary:")
    print(f"Generated non-special tokens: {len(result.generated_tokens)}")
    print(f"Mean log-probability: {result.mean_logprob}")
    print(f"Minimum log-probability: {result.min_logprob}")

    print("\nStage 1 smoke test finished successfully.")


if __name__ == "__main__":
    main()
