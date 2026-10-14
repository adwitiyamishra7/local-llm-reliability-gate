from scripts.benchmark_300_v4 import label_correct
from scripts.benchmark_provenance import audit_reference_provenance
import pandas as pd
from pathlib import Path


def test_numeric_reference_matching_preserves_signs():
    assert label_correct("The answer is -7.", "-7") == 1
    assert label_correct("The answer is -7.", "7") == 0
    assert label_correct("It is -7 times 6, or -42.", "-42") == 1
    assert label_correct("-7 multiplied by 6 is 42.", "-42") == 0


def test_numeric_reference_matching_accepts_equivalent_decimal_spelling():
    assert label_correct("Pure water has pH 7.0.", "7|seven") == 1
    assert label_correct("The value is 17.", "7") == 0


def test_reference_matching_handles_subscripts_and_latex_math():
    assert label_correct("The formula is H₂O.", "h2o") == 1
    assert label_correct(r"The complexity is $O(\log n)$.", "o(log n)|log n") == 1


def test_punctuation_only_reference_alias_is_literal():
    assert label_correct('A single-line comment starts with "#".', "#|hash") == 1
    assert label_correct(': "("#"', "#|hash") == 1
    assert label_correct("The symbol is @.", "#|hash") == 0


def test_canonical_question_and_reference_provenance_is_exact():
    source = pd.read_csv(Path("data/sample/benchmark_questions_300.csv"))
    benchmark = source.rename(columns={"id": "question_id", "expected": "reference"})
    digest, errors = audit_reference_provenance(benchmark)
    assert len(digest) == 64
    assert errors == []

    benchmark.loc[0, "reference"] = "tampered"
    _, errors = audit_reference_provenance(benchmark)
    assert any("q001: reference differs" in error for error in errors)
