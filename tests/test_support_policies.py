import unittest

from src.app.domain.support.models import KnowledgeSection, SupportKnowledge
from src.app.domain.support.policies import get_policy_entries, has_policy_content


class SupportPolicyTests(unittest.TestCase):
    def test_get_policy_entries_returns_escalation_rules_when_present(self) -> None:
        knowledge = SupportKnowledge(
            sections=(
                KnowledgeSection(name="FAQs", entries=("Shipping takes 3 days.",)),
                KnowledgeSection(
                    name="Policies",
                    entries=(
                        "Escalate account access issues.",
                        "Never approve refunds without policy support.",
                    ),
                ),
            )
        )

        self.assertEqual(
            get_policy_entries(knowledge),
            (
                "Escalate account access issues.",
                "Never approve refunds without policy support.",
            ),
        )
        self.assertTrue(has_policy_content(knowledge))

    def test_policy_lookup_is_case_insensitive_and_deterministic(self) -> None:
        knowledge = SupportKnowledge(
            sections=(
                KnowledgeSection(name="policies", entries=("Use lowercase match.",)),
                KnowledgeSection(name="Policies", entries=("Ignore later match.",)),
            )
        )

        self.assertEqual(get_policy_entries(knowledge), ("Use lowercase match.",))
        self.assertEqual(get_policy_entries(knowledge), ("Use lowercase match.",))

    def test_non_policy_content_does_not_trigger_policy_checks(self) -> None:
        knowledge = SupportKnowledge(
            sections=(
                KnowledgeSection(name="FAQs", entries=("Explain appointment booking.",)),
                KnowledgeSection(name="Escalations", entries=("Reference only.",)),
            )
        )

        self.assertEqual(get_policy_entries(knowledge), ())
        self.assertFalse(has_policy_content(knowledge))

    def test_empty_or_unsupported_policy_sections_are_handled_consistently(self) -> None:
        self.assertEqual(get_policy_entries(SupportKnowledge()), ())
        self.assertFalse(has_policy_content(SupportKnowledge()))
        self.assertEqual(
            get_policy_entries(
                SupportKnowledge(
                    sections=(KnowledgeSection(name="Policies", entries=()),)
                )
            ),
            (),
        )
