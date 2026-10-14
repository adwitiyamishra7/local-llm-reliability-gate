from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.gate.agreement import agreement_score, consensus_answer
from src.gate.generation import generate_one
from src.gate.model import load_local_llm


QUESTION = "What is the capital of Australia?"
SAMPLES = 5


def main() -> None:

    print("=" * 60)
    print("LOCAL LLM RELIABILITY GATE - CHECKPOINT 2")
    print("FIVE-SAMPLE SELF-CONSISTENCY TEST")
    print("=" * 60)

    llm = load_local_llm()

    print("\nQuestion:")
    print(QUESTION)

    results = []

    total_start = perf_counter()

    for index in range(SAMPLES):

        print(
            f"\nGenerating answer "
            f"{index + 1}/{SAMPLES}..."
        )

        result = generate_one(
            llm,
            question=QUESTION,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        )

        results.append(result)

        print(
            f"Answer {index + 1}: "
            f"{result.answer}"
        )

        print(
            f"Mean token log-probability: "
            f"{result.mean_logprob:.6f}"
            if result.mean_logprob is not None
            else "Mean token log-probability: N/A"
        )

        print(
            f"Minimum token log-probability: "
            f"{result.min_logprob:.6f}"
            if result.min_logprob is not None
            else "Minimum token log-probability: N/A"
        )

        print(
            f"Latency: "
            f"{result.latency_seconds:.3f} seconds"
        )

    total_latency = perf_counter() - total_start

    answers = [
        result.answer
        for result in results
    ]

    agreement = agreement_score(answers)
    consensus = consensus_answer(answers)

    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY RESULT")
    print("=" * 60)

    for index, answer in enumerate(
        answers,
        start=1,
    ):
        print(
            f"{index}. {answer}"
        )

    print(
        f"\nConsensus answer: "
        f"{consensus}"
    )

    print(
        f"Agreement score: "
        f"{agreement:.2f} "
        f"({agreement * SAMPLES:.0f}/{SAMPLES})"
    )

    print(
        f"Total latency: "
        f"{total_latency:.3f} seconds"
    )

    print(
        f"Average latency/sample: "
        f"{total_latency / SAMPLES:.3f} seconds"
    )

    print(
        "\nThis is an experiment, "
        "not a correctness guarantee."
    )


if __name__ == "__main__":
    main()