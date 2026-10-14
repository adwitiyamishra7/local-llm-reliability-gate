from __future__ import annotations

"""Lightweight response-usefulness guard for the live UI.

This is a post-classifier safety rule. It does not alter the trained model,
its 15 input features, its calibration, or the stored probability estimate.
It only catches obvious non-answers such as echoing the question's subject.
It does NOT verify factual correctness.
"""

import re
from dataclasses import dataclass


_WORD_RE = re.compile(r"[a-z]+(?:['-][a-z]+)?|\d+(?:\.\d+)?%?", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "who", "what", "where", "when", "why", "how",
    "which", "whose", "whom", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "and", "or", "as", "it", "its", "this", "that", "these",
    "those", "me", "please", "tell", "explain", "describe", "define", "give",
    "can", "could", "would", "should", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "about", "into", "than", "then", "first",
}

_NON_ANSWER_PATTERNS = (
    re.compile(r"\b(i\s+)?(do not|don't|cannot|can't|could not|couldn't)\s+(know|tell|determine|answer)\b", re.I),
    re.compile(r"\b(not sure|uncertain|unknown|no idea)\b", re.I),
    re.compile(r"\bas an ai\b", re.I),
    re.compile(r"\b(insufficient information|not enough information)\b", re.I),
)


@dataclass(frozen=True)
class ResponseQuality:
    is_useful: bool
    reason: str


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("'-") for token in _WORD_RE.findall(str(text or ""))]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token and token not in _STOP_WORDS]


def _compact(text: str) -> str:
    return " ".join(_tokens(text))


def assess_response_quality(question: str, answer: str) -> ResponseQuality:
    """Flag obvious empty, refusal-style, or question-echo responses.

    This is deliberately conservative: a PASS only means the answer did not
    trigger these simple usefulness checks. It does not mean the answer is true.
    """
    question = str(question or "").strip()
    answer = str(answer or "").strip()

    if not answer:
        return ResponseQuality(False, "No answer text was generated.")

    cleaned_answer = re.sub(r"^(the\s+answer\s+is|answer\s*:|it\s+is|it's)\s+", "", answer, flags=re.I).strip()
    compact_answer = _compact(cleaned_answer)

    if not compact_answer:
        return ResponseQuality(False, "The generated response contains no usable answer text.")

    for pattern in _NON_ANSWER_PATTERNS:
        if pattern.search(answer):
            return ResponseQuality(
                False,
                "The model expressed uncertainty or declined to answer instead of providing a response.",
            )

    compact_question = _compact(question)
    if compact_answer == compact_question:
        return ResponseQuality(
            False,
            "The response repeats the question instead of answering it.",
        )

    answer_content = _content_tokens(cleaned_answer)
    question_content = set(_content_tokens(question))

    # Catch short subject echoes: “Who is Ratan Tata?” -> “Ratan Tata.”,
    # “Who wrote Hamlet?” -> “Hamlet”, or “What is the capital of Australia?” -> “Australia”.
    # Requiring every meaningful answer token to appear in the question reduces
    # the risk of flagging a real answer that introduces new information.
    if answer_content and len(answer_content) <= 5:
        if all(token in question_content for token in answer_content):
            return ResponseQuality(
                False,
                "The response only repeats terms from the question and does not provide the requested information.",
            )

    # Catch answers that are essentially just punctuation or a generic label.
    if compact_answer in {"answer", "the answer", "result", "the result", "question"}:
        return ResponseQuality(
            False,
            "The model returned a generic label rather than an actual answer.",
        )

    return ResponseQuality(True, "Passed the basic response-usefulness checks.")
