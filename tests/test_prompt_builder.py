import unittest

from src.app.domain.support.models import ConversationTurn, PromptBuildInput
from src.app.domain.support.prompt_builder import SupportPromptBuilder


class SupportPromptBuilderTests(unittest.TestCase):
    def test_build_returns_support_system_and_user_prompts(self) -> None:
        builder = SupportPromptBuilder()

        prompt = builder.build(
            PromptBuildInput(
                history=[
                    ConversationTurn(role="user", content="Where is my order?"),
                    ConversationTurn(
                        role="assistant",
                        content="Can you share your order number?",
                    ),
                ],
                user_message="It is order 1234.",
            )
        )

        self.assertIn("business support assistant", prompt.system_prompt)
        self.assertIn("Tone:", prompt.system_prompt)
        self.assertIn("Operating rules:", prompt.system_prompt)
        self.assertIn("Constraints:", prompt.system_prompt)
        self.assertIn("User: Where is my order?", prompt.user_prompt)
        self.assertIn(
            "Assistant: Can you share your order number?",
            prompt.user_prompt,
        )
        self.assertIn("Latest customer message:\nIt is order 1234.", prompt.user_prompt)

    def test_build_handles_empty_history(self) -> None:
        builder = SupportPromptBuilder()

        prompt = builder.build(PromptBuildInput(user_message="Hello"))

        self.assertIn("No previous conversation history.", prompt.user_prompt)
