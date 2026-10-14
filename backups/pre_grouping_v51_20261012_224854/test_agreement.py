from src.gate.agreement import agreement_score, consensus_answer, normalize_answer


def test_normalize_answer():
    assert normalize_answer("  **Canberra.**  ") == "canberra"


def test_full_agreement():
    assert agreement_score(
        ["Canberra", "canberra", "Canberra.", "CANBERRA", "Canberra"]
    ) == 1.0


def test_partial_agreement():
    assert agreement_score(["A", "A", "B", "A", "C"]) == 0.6


def test_consensus_answer():
    assert consensus_answer(
        ["Sydney", "Canberra", "Canberra", "Canberra", "Sydney"]
    ) == "Canberra"
