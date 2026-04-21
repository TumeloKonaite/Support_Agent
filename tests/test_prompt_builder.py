import json
import unittest

from src.app.domain.support.models import (
    BusinessProfile,
    ConversationTurn,
    KnowledgeSection,
    PromptBuildInput,
    SupportKnowledge,
)
from src.app.domain.support.observability import SupportObservabilitySettings
from src.app.domain.support.prompt_builder import SupportPromptBuilder


class StaticBusinessProfileSource:
    def __init__(self, profile: BusinessProfile) -> None:
        self._profile = profile
        self.tenant_ids: list[str | None] = []

    def load(self, tenant_id: str | None = None) -> BusinessProfile:
        self.tenant_ids.append(tenant_id)
        return self._profile


class StaticKnowledgeSource:
    def __init__(self, knowledge: SupportKnowledge) -> None:
        self._knowledge = knowledge
        self.tenant_ids: list[str | None] = []

    def load(self, tenant_id: str | None = None) -> SupportKnowledge:
        self.tenant_ids.append(tenant_id)
        return self._knowledge


class SupportPromptBuilderTests(unittest.TestCase):
    def _parse_log_record(self, record: str) -> dict[str, object]:
        return json.loads(record)

    def _build_builder(
        self,
        observability: SupportObservabilitySettings | None = None,
    ) -> SupportPromptBuilder:
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
            ),
            observability=observability,
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

    def test_build_includes_retrieved_context_when_available(self) -> None:
        builder = self._build_builder()

        prompt = builder.build(
            PromptBuildInput(
                user_message="Can I get a refund?",
                retrieved_context=(
                    "Source: data/knowledge.json\nContent: Refunds over 30 days require human review.",
                ),
            )
        )

        self.assertIn("Retrieved business context:", prompt.user_prompt)
        self.assertIn("Refunds over 30 days require human review.", prompt.user_prompt)

    def test_build_limits_and_truncates_retrieved_context(self) -> None:
        builder = self._build_builder()
        long_item = "A" * 600

        prompt = builder.build(
            PromptBuildInput(
                user_message="Tell me about policies",
                retrieved_context=(
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    long_item,
                ),
            )
        )

        self.assertEqual(prompt.user_prompt.count("\n- "), 5)
        self.assertNotIn("six", prompt.user_prompt)
        self.assertNotIn(long_item, prompt.user_prompt)

    def test_build_passes_tenant_id_to_content_sources(self) -> None:
        profile_source = StaticBusinessProfileSource(
            BusinessProfile(
                business_name="Glow Studio",
                assistant_identity="the Glow Studio support assistant",
            )
        )
        knowledge_source = StaticKnowledgeSource(SupportKnowledge())
        builder = SupportPromptBuilder(
            business_profile_source=profile_source,
            knowledge_source=knowledge_source,
        )

        builder.build(PromptBuildInput(user_message="Hello", tenant_id="tenant-a"))

        self.assertEqual(profile_source.tenant_ids, ["tenant-a"])
        self.assertEqual(knowledge_source.tenant_ids, ["tenant-a"])

    def test_build_omits_optional_sections_when_inputs_are_missing(self) -> None:
        builder = SupportPromptBuilder(
            business_profile_source=StaticBusinessProfileSource(
                BusinessProfile(
                    business_name="Bare Beauty",
                    assistant_identity="the Bare Beauty assistant",
                )
            ),
            knowledge_source=StaticKnowledgeSource(SupportKnowledge()),
        )

        prompt = builder.build(PromptBuildInput(user_message="Do you ship?"))

        self.assertEqual(
            prompt.system_prompt,
            "You are the Bare Beauty assistant for Bare Beauty.",
        )
        self.assertNotIn("Support contacts:", prompt.system_prompt)
        self.assertNotIn("Tone:", prompt.system_prompt)
        self.assertNotIn("Business knowledge:", prompt.system_prompt)

    def test_build_output_remains_stable_for_known_inputs(self) -> None:
        builder = self._build_builder()

        prompt = builder.build(
            PromptBuildInput(
                history=[
                    ConversationTurn(role="user", content="Can I return this?"),
                    ConversationTurn(role="assistant", content="What was opened?"),
                ],
                user_message="The serum box is sealed.",
                retrieved_context=(
                    "Source: data/knowledge.json\nContent: Sealed products can be returned within 30 days.",
                ),
            )
        )

        self.assertEqual(
            prompt.system_prompt,
            "\n\n".join(
                [
                    "You are the Glow Studio support assistant for Glow Studio.\n"
                    "Support hours: Weekdays 9 to 5\n"
                    "Escalation path: Escalate refunds and account access requests to the care team.\n"
                    "Business metadata:\n"
                    "- Region: US",
                    "Support contacts:\n"
                    "- Email: support@glow.example\n"
                    "- Phone: +1-555-0100",
                    "Tone:\n"
                    "- Be warm and practical.\n"
                    "- Keep answers concise.",
                    "Business knowledge:\n"
                    "Policies:\n"
                    "- Never invent refund approvals.\n"
                    "- Escalate account changes to a human agent.\n\n"
                    "FAQs:\n"
                    "- Customers often ask about appointments.",
                ]
            ),
        )
        self.assertEqual(
            prompt.user_prompt,
            "Support conversation context:\n"
            "User: Can I return this?\n"
            "Assistant: What was opened?\n\n"
            "Retrieved business context:\n"
            "- Source: data/knowledge.json Content: Sealed products can be returned within 30 days.\n\n"
            "Latest customer message:\n"
            "The serum box is sealed.\n\n"
            "Respond as the support assistant.",
        )

    def test_build_logs_prompt_assembly_metadata_with_redacted_preview(self) -> None:
        builder = self._build_builder(
            SupportObservabilitySettings(
                prompt_preview_enabled=True,
                redact_sensitive_fields=True,
                max_preview_chars=120,
            )
        )

        with self.assertLogs("src.app.domain.support.prompt_builder", level="INFO") as logs:
            builder.build(
                PromptBuildInput(
                    history=[ConversationTurn(role="user", content="Order 12345")],
                    user_message="My email is customer@example.com and order 12345 is late.",
                    request_id="session-1",
                    retrieved_context=(
                        "Source: data/knowledge.json\nContent: Orders over 1000 require manual review.",
                    ),
                )
            )

        event = self._parse_log_record(logs.records[-1].getMessage())
        self.assertEqual(event["event"], "support.prompt.assembled")
        self.assertEqual(event["request_id"], "session-1")
        self.assertEqual(event["included_retrieval_context"], True)
        self.assertEqual(event["retrieved_chunk_count"], 1)
        self.assertEqual(
            event["system_sections"],
            ["identity", "contact", "tone", "knowledge"],
        )
        self.assertEqual(
            event["user_sections"],
            [
                "conversation_context",
                "retrieved_business_context",
                "latest_customer_message",
            ],
        )
        self.assertEqual(
            event["user_message"]["preview"],
            "My email is [redacted-email] and order [redacted-number] is late.",
        )
        self.assertEqual(
            event["retrieved_context_previews"][0]["preview"],
            "Source: data/knowledge.json Content: Orders over [redacted-number] require manual review.",
        )
