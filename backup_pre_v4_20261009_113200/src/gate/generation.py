from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from .model import LocalLLM


@dataclass
class GenerationResult:
    question: str
    answer: str
    generated_tokens: list[str]
    token_logprobs: list[float]
    mean_logprob: float | None
    min_logprob: float | None
    latency_seconds: float


def build_prompt(
    tokenizer,
    question: str,
    enable_thinking: bool = False,
) -> str:
    """
    Build a concise factual QA prompt.

    Important behavior:
    - Answer the question's intent, not merely repeat its subject.
    - "Who is/was X?" expects a short identifying sentence.
    - "What is X?" expects a definition or identification.
    - Direct factoid/calculation questions should still receive direct answers.
    """

    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise factual question-answering assistant. "
                "Answer the exact question with the information requested; "
                "do not merely repeat a name or the words in the question. "
                "For a question like 'Who is [person]?', identify the person "
                "in one short sentence, including their role or what they are "
                "known for, rather than returning the name alone. "
                "For a question like 'What is [thing]?', briefly define or "
                "identify the thing. For calculations, clearly give the result. "
                "For questions asking only for a name, date, number, or place, "
                "give that value directly. Usually use one concise sentence. "
                "Do not use markdown or show step-by-step reasoning. "
                "If you genuinely do not know, say 'I don't know' rather than "
                "inventing an answer."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def generate_one(
    llm: LocalLLM,
    question: str,
    max_new_tokens: int = 16,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
) -> GenerationResult:
    """
    Generate one answer and collect token log-probabilities from the model's
    original, unprocessed logits.

    The default generation settings and probability calculations are preserved.
    Only the prompt's answer-quality instructions have been improved.
    """

    if not question or not question.strip():
        raise ValueError("Question must not be empty.")

    prompt = build_prompt(
        llm.tokenizer,
        question,
        enable_thinking=False,
    )

    inputs = llm.tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(llm.device)
        for key, value in inputs.items()
    }

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_dict_in_generate": True,
        "output_logits": True,
    }

    if llm.tokenizer.pad_token_id is not None:
        generation_kwargs["pad_token_id"] = llm.tokenizer.pad_token_id

    if do_sample:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )

    if llm.device.type == "cuda":
        torch.cuda.synchronize()

    start = perf_counter()

    with torch.inference_mode():
        outputs = llm.model.generate(
            **inputs,
            **generation_kwargs,
        )

    if llm.device.type == "cuda":
        torch.cuda.synchronize()

    latency = perf_counter() - start

    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[:, input_length:]
    ids = generated_ids[0].detach().cpu().tolist()

    special_ids = set(llm.tokenizer.all_special_ids)

    token_strings: list[str] = []
    token_logprobs: list[float] = []

    raw_logits_steps = outputs.logits or ()

    for token_id, raw_logits in zip(ids, raw_logits_steps):
        if token_id in special_ids:
            continue

        log_probs = torch.log_softmax(
            raw_logits[0].float(),
            dim=-1,
        )

        token_logprob = float(
            log_probs[token_id].detach().cpu()
        )

        if not np.isfinite(token_logprob):
            continue

        token_strings.append(
            llm.tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
            )
        )
        token_logprobs.append(token_logprob)

    clean_ids = [
        token_id
        for token_id in ids
        if token_id not in special_ids
    ]

    answer = llm.tokenizer.decode(
        clean_ids,
        skip_special_tokens=True,
    ).strip()

    mean_logprob = (
        float(np.mean(token_logprobs))
        if token_logprobs
        else None
    )

    min_logprob = (
        float(np.min(token_logprobs))
        if token_logprobs
        else None
    )

    return GenerationResult(
        question=question,
        answer=answer,
        generated_tokens=token_strings,
        token_logprobs=token_logprobs,
        mean_logprob=mean_logprob,
        min_logprob=min_logprob,
        latency_seconds=latency,
    )


def result_to_dict(
    result: GenerationResult,
) -> dict:
    """Convert generation result to a dictionary."""
    return asdict(result)
