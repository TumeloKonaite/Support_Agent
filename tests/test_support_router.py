import unittest

from src.app.domain.support.router import RouteType, RuleBasedSupportRouter


class SupportRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = RuleBasedSupportRouter()

    def test_decide_routes_to_rag_for_knowledge_queries(self) -> None:
        decision = self.router.decide("What is your returns policy?")

        self.assertEqual(decision.route, RouteType.RAG)
        self.assertEqual(decision.reason, "knowledge_lookup_required")

    def test_decide_routes_to_conversation_for_small_talk(self) -> None:
        decision = self.router.decide("Hello there")

        self.assertEqual(decision.route, RouteType.CONVERSATION)
        self.assertEqual(decision.reason, "conversational_message_detected")

    def test_decide_routes_to_tool_for_action_requests(self) -> None:
        decision = self.router.decide("Cancel my subscription and send me confirmation.")

        self.assertEqual(decision.route, RouteType.TOOL)
        self.assertEqual(decision.reason, "action_request_detected")

    def test_decide_keeps_how_do_i_questions_on_knowledge_path(self) -> None:
        decision = self.router.decide("How do I cancel my subscription?")

        self.assertEqual(decision.route, RouteType.RAG)

    def test_decide_routes_general_help_to_conversation(self) -> None:
        decision = self.router.decide("Can you introduce yourself?")

        self.assertEqual(decision.route, RouteType.CONVERSATION)
        self.assertEqual(decision.reason, "general_assistance_response")
