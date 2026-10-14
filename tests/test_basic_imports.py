def test_project_imports():
    from src.gate.model import DEFAULT_MODEL_NAME, get_device
    from src.gate.generation import GenerationResult, build_prompt
    from src.gate.agreement import agreement_score

    assert DEFAULT_MODEL_NAME == "Qwen/Qwen3-0.6B"
    assert callable(get_device)
    assert GenerationResult is not None
    assert callable(build_prompt)
    assert callable(agreement_score)
