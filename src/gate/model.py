from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"

_WARMUP_LOCK = Lock()
_WARMUP_MODEL = None
_WARMUP_DONE = False


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


def warm_up(model: LocalLLM | None = None) -> LocalLLM:
    """Load (and cache) the local model if needed, then run a one-token warm-up once."""
    global _WARMUP_MODEL, _WARMUP_DONE

    with _WARMUP_LOCK:
        target = model
        if target is None:
            if _WARMUP_MODEL is None:
                _WARMUP_MODEL = load_local_llm()
            target = _WARMUP_MODEL
            if _WARMUP_DONE:
                return target
        elif getattr(target, "_gate_warmed_up", False):
            return target

        # Lazy import avoids the model <-> generation import cycle.
        from src.gate.generation import generate_one

        generate_one(
            target,
            "Warm-up.",
            max_new_tokens=1,
            do_sample=False,
        )

        if model is None:
            _WARMUP_DONE = True
        else:
            setattr(target, "_gate_warmed_up", True)
        return target


def get_device_label() -> str:
    """Return a short display label for the selected runtime device."""
    device = get_device()
    if device.type != "cuda":
        return "CPU"
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "GPU"
    return f"CUDA · {gpu_name}"
