import unittest

from src.app.domain.support.models import SupportContextChunk
from src.app.domain.support.retrieval import RetrievalDecision
from src.app.domain.support.router import RouteType, RuleBasedSupportRouter


class SupportRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = RuleBasedSupportRouter()

    def test_decide_routes_to_rag_for_grounded_knowledge_queries(self) -> None:
        decision = self.router.decide(
            "What is your returns policy?",
            RetrievalDecision(
                retrieved_context=(
                    SupportContextChunk(
                        chunk_id="chunk-1",
                        label="[1]",
                        text="Returns are accepted within 30 days.",
                        source="knowledge.json",
                        score=0.91,
                    ),
                ),
                confidence_score=0.91,
                used_fallback=False,
                decision_reason="high_confidence",
            ),
        )

        self.assertEqual(decision.route, RouteType.RAG)
        self.assertEqual(decision.reason, "grounded_retrieval_available")
        self.assertAlmostEqual(decision.confidence or 0.0, 0.91)

    def test_decide_routes_to_fallback_for_weak_retrieval(self) -> None:
        decision = self.router.decide(
            "What do you think about the weather today?",
            RetrievalDecision(
                confidence_score=0.12,
                used_fallback=True,
                decision_reason="low_confidence",
            ),
        )

        self.assertEqual(decision.route, RouteType.FALLBACK)
        self.assertEqual(decision.reason, "low_confidence")
        self.assertAlmostEqual(decision.confidence or 0.0, 0.12)

    def test_decide_routes_to_tool_for_action_requests(self) -> None:
        decision = self.router.decide(
            "Cancel my subscription and send me confirmation.",
            RetrievalDecision(
                confidence_score=0.88,
                used_fallback=False,
                decision_reason="high_confidence",
            ),
        )

        self.assertEqual(decision.route, RouteType.TOOL)
        self.assertEqual(decision.reason, "action_request_detected")
        self.assertAlmostEqual(decision.confidence or 0.0, 0.88)

    def test_decide_keeps_how_do_i_questions_on_knowledge_path(self) -> None:
        decision = self.router.decide(
            "How do I cancel my subscription?",
            RetrievalDecision(
                retrieved_context=(
                    SupportContextChunk(
                        chunk_id="chunk-2",
                        label="[1]",
                        text="Customers can cancel from the billing settings page.",
                        source="knowledge.json",
                        score=0.84,
                    ),
                ),
                confidence_score=0.84,
                used_fallback=False,
                decision_reason="high_confidence",
            ),
        )

        self.assertEqual(decision.route, RouteType.RAG)
