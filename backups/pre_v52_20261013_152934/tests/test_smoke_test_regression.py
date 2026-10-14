from scripts.smoke_test_real_model import evaluate_smoke_case, is_abstention_answer


def test_custom_question_without_expected_term_checks_usefulness_only():
    passed, reason = evaluate_smoke_case(
        answer="Isaac Newton was an English physicist.",
        quality_ok=True,
        must_be_useful=True,
        expected_needle=None,
    )
    assert passed is True
    assert reason == "quality_pass_accuracy_not_asserted"


def test_expected_term_mismatch_fails():
    passed, reason = evaluate_smoke_case(
        answer="The capital of Australia is Sydney.",
        quality_ok=True,
        must_be_useful=True,
        expected_needle="Canberra",
    )
    assert passed is False
    assert reason == "expected_term_missing"


def test_quality_failure_fails_even_if_expected_term_appears():
    passed, reason = evaluate_smoke_case(
        answer="I don't know, Canberra.",
        quality_ok=False,
        must_be_useful=True,
        expected_needle="Canberra",
    )
    assert passed is False
    assert reason == "quality_check_failed"


def test_unanswerable_question_requires_explicit_abstention():
    answer = "The name of my neighbour's cat is not provided."
    assert is_abstention_answer(answer)
    assert evaluate_smoke_case(answer, True, False, None) == (True, "abstention_detected")
    assert evaluate_smoke_case("The cat's name is Fluffy.", True, False, None) == (
        False, "expected_abstention_not_detected"
    )


def test_default_capital_expectation_detects_wrong_answer():
    passed, reason = evaluate_smoke_case(
        answer="The capital of Australia is Sydney.",
        quality_ok=True,
        must_be_useful=True,
        expected_needle="Canberra",
    )
    assert (passed, reason) == (False, "expected_term_missing")
