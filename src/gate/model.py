from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


@dataclass
class LocalLLM:
    """Loaded local model, tokenizer and runtime device."""

    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Choose CUDA when available; otherwise use CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_local_llm(
    model_name: str = DEFAULT_MODEL_NAME,
    prefer_gpu: bool = True,
) -> LocalLLM:
    """Load Qwen3-0.6B for the local reliability experiment."""
    device = get_device(prefer_gpu)

    if device.type == "cuda":
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    print(f"Runtime device: {device}")
    print(f"Model dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: CPU")

    return LocalLLM(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
