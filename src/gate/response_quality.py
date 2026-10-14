from __future__ import annotations

"""Lightweight response-usefulness guard (v2).

A PASS only means no obvious response-quality failure was detected (empty output,
refusal / non-answer, question echo, malformed or visibly truncated text). It is NOT
evidence that the answer is factually correct.

This module has no torch/transformers dependency so the UI and unit tests can import it.
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

# First-person refusals / lack-of-knowledge statements. First person is required for the
# "do not know" family so a valid answer like "historians do not know his birth date" is not
# rejected.
_REFUSAL_PATTERNS = (
    re.compile(r"\bI\s+(?:do\s+not|don['’]t|cannot|can['’]t|could\s+not|couldn['’]t)\s+(?:really\s+|actually\s+)?(?:know|tell|determine|answer|say|identify)\b", re.I),
    re.compile(r"\bI(?:['’]m|\s+am)\s+(?:not\s+sure|unsure|not\s+familiar|unfamiliar)\b", re.I),
    re.compile(r"\bI\s+(?:do\s+not|don['’]t)\s+have\s+(?:any\s+|enough\s+|specific\s+|sufficient\s+)?(?:information|knowledge|data|details)\b", re.I),
    re.compile(r"\bI\s+have\s+no\s+(?:information|knowledge|idea)\b", re.I),
    re.compile(r"\bas\s+an\s+ai\b", re.I),
    re.compile(r"\b(?:insufficient|not\s+enough)\s+information\b", re.I),
)
_LEADING_REFUSAL = re.compile(
    r"^\s*(?:i(?:['’]m|\s+am)\s+sorry\b|unknown\b|no\s+idea\b|not\s+sure\b|uncertain\b|n/?a\b)", re.I
)
_SHORT_UNCERTAIN = re.compile(r"\b(?:not\s+sure|uncertain|unknown|no\s+idea)\b", re.I)

_DANGLING_WORDS = {
    "is", "are", "was", "were", "and", "or", "of", "to", "the", "an", "in", "for",
    "with", "by", "as", "that", "which", "than", "because", "who", "whom", "whose",
    "where", "when", "but", "from", "at", "on", "about", "into", "also", "has", "have", "had",
}
_SENTENCE_END = (".", "!", "?", ")", '"', "'", "]", "”", "’")


@dataclass(frozen=True)
class ResponseQuality:
    is_useful: bool
    reason: str


def clean_generated_text(text: str) -> str:
    """Remove hidden-reasoning markup from decoded model text.

    Qwen3's <think> tokens are not flagged as special tokens, so they survive
    ``skip_special_tokens=True``. A complete <think>...</think> block is dropped; an
    unterminated <think> means the output is malformed and yields "" (a quality failure).
    """
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return ""
    return text.strip()


def _tokens(text: str) -> list[str]:
    return [token.lower().strip("'-") for token in _WORD_RE.findall(str(text or ""))]


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token and token not in _STOP_WORDS]


def _compact(text: str) -> str:
    return " ".join(_tokens(text))


def _looks_truncated(answer: str) -> bool:
    text = answer.rstrip()
    if not text:
        return True
    if text.count("$") % 2 == 1:
        return True
    if text.endswith((":", ",", "(", "-", "=", "×", "\\")):
        return True
    if text.endswith(_SENTENCE_END):
        return False
    last_word = re.findall(r"[A-Za-z0-9']+", text)
    return bool(last_word) and last_word[-1].lower() in _DANGLING_WORDS


def assess_response_quality(question: str, answer: str) -> ResponseQuality:
    """Flag obvious empty, refusal-style, echo, malformed or truncated responses."""
    question = str(question or "").strip()
    answer = str(answer or "").strip()

    if not answer:
        return ResponseQuality(False, "No answer text was generated.")

    if "<think>" in answer or "</think>" in answer or "<|" in answer or "\ufffd" in answer:
        return ResponseQuality(False, "The output contains hidden-reasoning markup, special tokens or invalid characters.")

    cleaned_answer = re.sub(r"^(the\s+answer\s+is|answer\s*:|it\s+is|it's)\s+", "", answer, flags=re.I).strip()
    compact_answer = _compact(cleaned_answer)

    if not compact_answer:
        return ResponseQuality(False, "The generated response contains no usable answer text.")

    refusal_reason = "The model expressed uncertainty or declined to answer instead of providing a response."
    if _LEADING_REFUSAL.search(answer) or any(p.search(answer) for p in _REFUSAL_PATTERNS):
        return ResponseQuality(False, refusal_reason)
    if len(answer.split()) <= 6 and _SHORT_UNCERTAIN.search(answer):
        return ResponseQuality(False, refusal_reason)

    if compact_answer == _compact(question):
        return ResponseQuality(False, "The response repeats the question instead of answering it.")

    answer_content = _content_tokens(cleaned_answer)
    question_content = set(_content_tokens(question))

    # Short subject echoes: "Who is Ratan Tata?" -> "Ratan Tata.", "Who wrote Hamlet?" -> "Hamlet".
    # Every meaningful answer token must come from the question, so real answers that add
    # information ("Canberra", "William Shakespeare", "36") pass.
    if answer_content and len(answer_content) <= 5:
        if all(token in question_content for token in answer_content):
            return ResponseQuality(
                False,
                "The response only repeats terms from the question and does not provide the requested information.",
            )

    if compact_answer in {"answer", "the answer", "result", "the result", "question"}:
        return ResponseQuality(False, "The model returned a generic label rather than an actual answer.")

    if _looks_truncated(answer):
        return ResponseQuality(False, "The response appears truncated or malformed (it ends mid-phrase).")

    return ResponseQuality(True, "Passed the basic response-usefulness checks (not a factual-accuracy check).")
