import unittest

from src.app.domain.support.guardrails import SAFE_FALLBACK_MESSAGE, SupportGuardrailPolicy
from src.app.domain.support.models import SupportContextChunk
from src.app.domain.support.retrieval import RetrievalDecision


class SupportGuardrailPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = SupportGuardrailPolicy()

    def test_evaluate_returns_grounded_answer_with_citations(self) -> None:
        decision = self.policy.evaluate(
            RetrievalDecision(
                retrieved_context=(
                    SupportContextChunk(
                        chunk_id="chunk-1",
                        label="[1]",
                        text="Refunds are processed in five business days.",
                        source="refunds.md",
                        score=0.91,
                    ),
                ),
                confidence_score=0.91,
                used_fallback=False,
                decision_reason="high_confidence",
            )
        )

        self.assertFalse(decision.should_fallback)
        self.assertEqual(decision.answer.grounding_status, "grounded")
        self.assertTrue(decision.answer.used_context)
        self.assertEqual(decision.answer.citations[0].label, "[1]")
        self.assertEqual(decision.context_chunks[0].source, "refunds.md")

    def test_evaluate_returns_safe_fallback_for_missing_context(self) -> None:
        decision = self.policy.evaluate(
            RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="no_results",
            )
        )

        self.assertTrue(decision.should_fallback)
        self.assertEqual(decision.answer.grounding_status, "fallback")
        self.assertEqual(decision.answer.fallback_reason, "no_results")
        self.assertEqual(decision.answer.message, SAFE_FALLBACK_MESSAGE)
