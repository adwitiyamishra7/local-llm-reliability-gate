from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gate.response_quality import assess_response_quality


class ResponseQualityTests(unittest.TestCase):
    def test_subject_echoes_are_flagged(self):
        cases = [
            ("Who is Ratan Tata?", "Ratan Tata."),
            ("What is the capital of Australia?", "Australia."),
            ("Who wrote Hamlet?", "Hamlet."),
            ("Who discovered penicillin?", "Penicillin."),
            ("What is Python?", "Python."),
            ("What is the square root of 144?", "144."),
        ]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                self.assertFalse(assess_response_quality(question, answer).is_useful)

    def test_actual_answer_shapes_pass(self):
        cases = [
            ("What is the capital of Australia?", "Canberra."),
            ("Who wrote Hamlet?", "William Shakespeare."),
            ("What is 15% of 240?", "36."),
            ("Who is Ratan Tata?", "Ratan Tata was an Indian industrialist."),
            ("Who was the first president of the United States?", "George Washington."),
            ("What is Python?", "Python is a high-level programming language."),
        ]
        for question, answer in cases:
            with self.subTest(question=question, answer=answer):
                self.assertTrue(assess_response_quality(question, answer).is_useful)

    def test_empty_and_refusal_responses_are_flagged(self):
        self.assertFalse(assess_response_quality("Who is Ratan Tata?", "").is_useful)
        self.assertFalse(assess_response_quality("Who is Ratan Tata?", "I don't know.").is_useful)


if __name__ == "__main__":
    unittest.main(verbosity=2)
