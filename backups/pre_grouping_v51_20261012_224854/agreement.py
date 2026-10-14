from __future__ import annotations

import re
import unicodedata
from collections import Counter


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    # Markdown
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)

    # LaTeX wrappers
    text = text.replace("$$", " ")
    text = text.replace("$", " ")

    # Common LaTeX operators
    text = text.replace(r"\times", "×")
    text = text.replace(r"\cdot", "×")
    text = text.replace(r"\sqrt", "sqrt")
    text = text.replace(r"\circ", "°")
    text = text.replace(r"\degree", "°")

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# GENERAL NORMALIZATION
# ============================================================

def normalize_answer(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    text = text.lower()

    # Normalize apostrophes
    text = text.replace("’", "'")

    # Remove parenthetical abbreviations
    # Example:
    # "Newton (N)" -> "Newton"
    text = re.sub(
        r"\s*\([^)]*\)",
        "",
        text,
    )

    # Normalize J.K. Rowling
    text = re.sub(
        r"\bj\s*\.\s*k\s*\.\s*rowling\b",
        "jk rowling",
        text,
        flags=re.IGNORECASE,
    )

    # Remove punctuation around words
    text = text.strip(
        " \t\n\r.,;:!?\"'()[]{}"
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# FIRST SENTENCE
# ============================================================

def _first_sentence(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    match = re.search(
        r"^(.+?[.!?])(?:\s|$)",
        text,
    )

    if match:
        return match.group(1).strip()

    return text.strip()


# ============================================================
# KEY NORMALIZATION
# ============================================================

def normalize_extracted_key(
    text: str,
) -> str:

    text = clean_text(text)

    if not text:
        return ""

    # Remove unfinished LaTeX commands
    text = re.sub(
        r"\\[a-zA-Z]+",
        " ",
        text,
    )

    # Remove dollar signs
    text = text.replace("$", "")

    # Remove parenthetical abbreviations
    text = re.sub(
        r"\s*\([^)]*\)",
        "",
        text,
    )

    # Common unit normalization
    text = re.sub(
        r"\bnewtons?\b",
        "newton",
        text,
        flags=re.IGNORECASE,
    )

    # Remove leading article
    text = re.sub(
        r"^(the|a|an)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # J.K. Rowling
    text = re.sub(
        r"\bj\s*\.\s*k\s*\.\s*rowling\b",
        "jk rowling",
        text,
        flags=re.IGNORECASE,
    )

    # "painted by Leonardo da Vinci" -> "Leonardo da Vinci"
    text = re.sub(
        r"^painted by\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize "measured in Newtons" -> "Newton"
    text = re.sub(
        r"^measured in\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize "located in Egypt" -> "Egypt"
    text = re.sub(
        r"^located in\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove trailing punctuation
    text = text.strip(
        " \t\n\r.,;:!?\"'()[]{}"
    )

    # Normalize degree notation
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*degrees?\s*celsius",
        r"\1°C",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*°\s*C\b",
        r"\1°C",
        text,
        flags=re.IGNORECASE,
    )

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return normalize_answer(text)


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer_key(
    question: str,
    answer: str,
) -> str:

    q = normalize_answer(question)
    text = clean_text(answer)

    if not text:
        return ""

    # --------------------------------------------------------
    # Explicit answer cues
    # --------------------------------------------------------

    patterns = [
        r"\bwritten by\s+(.+)",
        r"\bpainted by\s+(.+)",
        r"\binvented by\s+(.+)",
        r"\bcreated by\s+(.+)",
        r"\bdiscovered by\s+(.+)",
        r"\bdeveloped by\s+(.+)",

        r"\bstands for\s+(.+)",
        r"\bstand for\s+(.+)",

        r"\blocated in\s+(.+)",
        r"\bknown as\s+(.+)",
        r"\bcalled\s+(.+)",

        r"\bmeasured in\s+(.+)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            value = match.group(1)

            value = re.split(
                r"[.!?\n]",
                value,
                maxsplit=1,
            )[0]

            key = normalize_extracted_key(
                value
            )

            if key:
                return key

    # --------------------------------------------------------
    # Equation-style answers
    # --------------------------------------------------------

    equation_match = re.search(
        r"=\s*([^\n]+)",
        text,
    )

    if equation_match:

        value = equation_match.group(1)

        value = re.split(
            r"[.!?\n]",
            value,
            maxsplit=1,
        )[0]

        key = normalize_extracted_key(
            value
        )

        if key:
            return key

    # --------------------------------------------------------
    # "is / are / was / were" answers
    # --------------------------------------------------------

    match = re.search(
        r"\b(?:is|are|was|were)\s+(?:the\s+)?(.+)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        value = match.group(1)

        value = re.split(
            r"[.!?\n]",
            value,
            maxsplit=1,
        )[0]

        key = normalize_extracted_key(
            value
        )

        # Avoid returning empty / meaningless fragments
        if key and key not in {
            "calculated",
            "calculated as",
            "the",
        }:
            return key

    # --------------------------------------------------------
    # Numeric questions
    # --------------------------------------------------------

    numeric_question_words = [
        "how many",
        "how much",
        "percentage",
        "percent",
        "square root",
        "multiplied",
        "divided",
        "average",
        "median",
        "area",
        "perimeter",
        "sum of",
        "year",
        "meters",
        "degrees",
    ]

    if any(
        word in q
        for word in numeric_question_words
    ):

        numbers = re.findall(
            r"-?\d+(?:,\d{3})*(?:\.\d+)?",
            text,
        )

        if numbers:

            value = numbers[-1]

            return normalize_extracted_key(
                value
            )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    sentence = _first_sentence(text)

    return normalize_extracted_key(
        sentence
    )


def answers_are_equivalent(
    question: str,
    left_answer: str,
    right_answer: str,
    left_key: str | None = None,
    right_key: str | None = None,
) -> bool:
    """Return True only for conservative, text-supported answer equivalence.

    This is not a semantic truth check.  It handles harmless presentation
    differences such as a bare answer versus an explanatory sentence containing
    that answer, and repeated answers to ``Who is X?`` that begin with X.  It
    deliberately does not infer that two unrelated candidates are equivalent.
    """
    left_key = extract_answer_key(question, left_answer) if left_key is None else left_key
    right_key = extract_answer_key(question, right_answer) if right_key is None else right_key

    if left_key == right_key:
        return True

    left_text = normalize_answer(left_answer)
    right_text = normalize_answer(right_answer)

    if not left_text or not right_text:
        return False

    # A direct answer and a longer explanation containing it are the same
    # candidate, regardless of where the extractor placed the sentence cue.
    if left_text in right_text or right_text in left_text:
        return True

    # The extractor may return the direct value from one sentence and a longer
    # phrase from another.  Only merge when the extracted value is visibly
    # present in the other answer; this is not a fuzzy semantic match.
    if left_key and left_key in right_text:
        return True
    if right_key and right_key in left_text:
        return True

    # For identity questions, repeated answers that start with the asked-about
    # subject are equivalent even when their explanatory tails differ.
    question_match = re.match(r"who\s+(?:is|was|are|were)\s+(.+)$", normalize_answer(question))
    if question_match:
        subject = question_match.group(1)
        if left_text.startswith(subject) and right_text.startswith(subject):
            return True

    return False


def grouped_answer_keys(question: str, answers: list[str]) -> list[str]:
    """Return representative keys after conservative equivalence grouping."""
    grouped: list[str] = []
    representatives: list[tuple[str, str]] = []

    for answer in answers:
        raw_key = extract_answer_key(question, answer)
        for index, (representative_answer, representative_key) in enumerate(representatives):
            if answers_are_equivalent(
                question,
                answer,
                representative_answer,
                raw_key,
                representative_key,
            ):
                grouped.append(representative_key)
                break
        else:
            representatives.append((answer, raw_key))
            grouped.append(raw_key)

    return grouped


# ============================================================
# AGREEMENT
# ============================================================

def agreement_score(
    answers: list[str],
    questions: list[str] | None = None,
) -> float:

    if not answers:
        return 0.0

    if questions is None:
        questions = [""] * len(answers)

    keys = [
        extract_answer_key(
            question,
            answer,
        )
        for answer, question
        in zip(answers, questions)
    ]

    counts = Counter(keys)

    return (
        counts.most_common(1)[0][1]
        / len(keys)
    )


# ============================================================
# CONSENSUS
# ============================================================

def consensus_answer(
    answers: list[str],
    questions: list[str] | None = None,
) -> str:

    if not answers:
        return ""

    if questions is None:
        questions = [""] * len(answers)

    keys = [
        extract_answer_key(
            question,
            answer,
        )
        for answer, question
        in zip(answers, questions)
    ]

    counts = Counter(keys)

    winning_key = (
        counts.most_common(1)[0][0]
    )

    for answer, key in zip(
        answers,
        keys,
    ):
        if key == winning_key:
            return answer

    return answers[0]
