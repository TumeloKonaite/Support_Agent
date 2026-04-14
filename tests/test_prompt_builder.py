import unittest

from src.app.domain.support.models import (
    BusinessProfile,
    ConversationTurn,
    KnowledgeSection,
    PromptBuildInput,
    SupportKnowledge,
)
from src.app.domain.support.prompt_builder import SupportPromptBuilder


class StaticBusinessProfileSource:
    def __init__(self, profile: BusinessProfile) -> None:
        self._profile = profile

    def load(self, tenant_id: str | None = None) -> BusinessProfile:
        return self._profile


class StaticKnowledgeSource:
    def __init__(self, knowledge: SupportKnowledge) -> None:
        self._knowledge = knowledge

    def load(self, tenant_id: str | None = None) -> SupportKnowledge:
        return self._knowledge


class SupportPromptBuilderTests(unittest.TestCase):
    def _build_builder(self) -> SupportPromptBuilder:
        return SupportPromptBuilder(
            business_profile_source=StaticBusinessProfileSource(
                BusinessProfile(
                    business_name="Glow Studio",
                    assistant_identity="the Glow Studio support assistant",
                    support_email="support@glow.example",
                    support_phone="+1-555-0100",
                    escalation_target="Escalate refunds and account access requests to the care team.",
                    support_hours="Weekdays 9 to 5",
                    tone_guidelines=(
                        "Be warm and practical.",
                        "Keep answers concise.",
                    ),
                    metadata={"region": "US"},
                ),
            ),
            knowledge_source=StaticKnowledgeSource(
                SupportKnowledge(
                    sections=(
                        KnowledgeSection(
                            name="Policies",
                            entries=(
                                "Never invent refund approvals.",
                                "Escalate account changes to a human agent.",
                            ),
                        ),
                        KnowledgeSection(
                            name="FAQs",
                            entries=("Customers often ask about appointments.",),
                        ),
                    )
                )
            )
        )

    def test_build_returns_support_system_and_user_prompts(self) -> None:
        builder = self._build_builder()

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

        self.assertIn("Glow Studio", prompt.system_prompt)
        self.assertIn("Support contacts:", prompt.system_prompt)
        self.assertIn("Tone:", prompt.system_prompt)
        self.assertIn("Business knowledge:", prompt.system_prompt)
        self.assertIn("Policies:", prompt.system_prompt)
        self.assertIn("User: Where is my order?", prompt.user_prompt)
        self.assertIn(
            "Assistant: Can you share your order number?",
            prompt.user_prompt,
        )
        self.assertIn("Latest customer message:\nIt is order 1234.", prompt.user_prompt)

    def test_build_handles_empty_history(self) -> None:
        builder = self._build_builder()

        prompt = builder.build(PromptBuildInput(user_message="Hello"))

        self.assertIn("No previous conversation history.", prompt.user_prompt)
