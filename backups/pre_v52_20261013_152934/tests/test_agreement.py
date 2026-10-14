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


def test_containment_does_not_merge_explicitly_negated_answer():
    from src.gate.agreement import answers_are_equivalent, grouped_answer_keys

    question = "What is the capital of Australia?"
    answer = "Canberra"
    contradiction = "Canberra is not the capital; Sydney is the capital."

    assert answers_are_equivalent(question, answer, "The capital of Australia is Canberra.")
    assert not answers_are_equivalent(question, answer, contradiction)
    assert grouped_answer_keys(question, [answer, contradiction])[0] != grouped_answer_keys(
        question, [answer, contradiction]
    )[1]


def test_same_person_prefix_does_not_hide_conflicting_years():
    from src.gate.agreement import answers_are_equivalent, grouped_answer_keys

    question = "Who is Isaac Newton?"
    claimed_1900 = "Isaac Newton was born in 1900."
    claimed_1643 = "Isaac Newton was born in 1643."

    assert not answers_are_equivalent(question, claimed_1900, claimed_1643)
    keys = grouped_answer_keys(question, [claimed_1900, claimed_1643])
    assert keys[0] != keys[1]
