from typing import Protocol

from src.app.domain.support.models import (
    BusinessProfile,
    ConversationTurn,
    PromptBuildInput,
    PromptBuildResult,
    SupportKnowledge,
)


class BusinessProfileSource(Protocol):
    """Runtime source for business profile content."""

    def load(self, tenant_id: str | None = None) -> BusinessProfile: ...


class KnowledgeSource(Protocol):
    """Runtime source for support knowledge content."""

    def load(self, tenant_id: str | None = None) -> SupportKnowledge: ...


class SupportPromptBuilder:
    """Build support-domain prompts for the LLM layer."""

    def __init__(
        self,
        business_profile_source: BusinessProfileSource,
        knowledge_source: KnowledgeSource,
    ) -> None:
        self._business_profile_source = business_profile_source
        self._knowledge_source = knowledge_source

    def build(self, prompt_input: PromptBuildInput) -> PromptBuildResult:
        """Generate the system and user prompts for a support interaction."""
        business_profile = self._business_profile_source.load(prompt_input.tenant_id)
        knowledge = self._knowledge_source.load(prompt_input.tenant_id)
        return PromptBuildResult(
            system_prompt=self._build_system_prompt(business_profile, knowledge),
            user_prompt=self._build_user_prompt(
                history=prompt_input.history,
                user_message=prompt_input.user_message,
                retrieved_context=prompt_input.retrieved_context,
            ),
        )

    def _build_system_prompt(
        self,
        business_profile: BusinessProfile,
        knowledge: SupportKnowledge,
    ) -> str:
        sections = [
            self._render_identity_section(business_profile),
            self._render_contact_section(business_profile),
            self._render_tone_section(business_profile),
            self._render_knowledge_sections(knowledge),
        ]
        return "\n\n".join(section for section in sections if section)

    def _build_user_prompt(
        self,
        history: list[ConversationTurn],
        user_message: str,
        retrieved_context: tuple[str, ...],
    ) -> str:
        history_text = self._render_history(history)
        sections = [
            "Support conversation context:\n"
            f"{history_text}"
        ]
        retrieved_context_text = self._render_retrieved_context(retrieved_context)
        if retrieved_context_text:
            sections.append(retrieved_context_text)
        sections.append(
            "Latest customer message:\n"
            f"{user_message}\n\n"
            "Respond as the support assistant."
        )
        return "\n\n".join(sections)

    def _render_history(self, history: list[ConversationTurn]) -> str:
        if not history:
            return "No previous conversation history."

        return "\n".join(f"{turn.role.title()}: {turn.content}" for turn in history)

    def _render_retrieved_context(self, retrieved_context: tuple[str, ...]) -> str:
        if not retrieved_context:
            return ""

        max_items = 5
        max_item_length = 500
        lines = ["Retrieved business context:"]
        for item in retrieved_context[:max_items]:
            normalized = " ".join(item.split())
            if len(normalized) > max_item_length:
                normalized = normalized[: max_item_length - 3].rstrip() + "..."
            lines.append(f"- {normalized}")
        return "\n".join(lines)

    def _render_identity_section(self, business_profile: BusinessProfile) -> str:
        lines = [
            (
                f"You are {business_profile.assistant_identity} for "
                f"{business_profile.business_name}."
            )
        ]

        if business_profile.support_hours:
            lines.append(f"Support hours: {business_profile.support_hours}")
        if business_profile.escalation_target:
            lines.append(f"Escalation path: {business_profile.escalation_target}")
        if business_profile.metadata:
            lines.append("Business metadata:")
            lines.extend(
                f"- {key.replace('_', ' ').title()}: {value}"
                for key, value in sorted(business_profile.metadata.items())
            )

        return "\n".join(lines)

    def _render_contact_section(self, business_profile: BusinessProfile) -> str:
        contacts: list[str] = []
        if business_profile.support_email:
            contacts.append(f"- Email: {business_profile.support_email}")
        if business_profile.support_phone:
            contacts.append(f"- Phone: {business_profile.support_phone}")

        if not contacts:
            return ""

        return "Support contacts:\n" + "\n".join(contacts)

    def _render_tone_section(self, business_profile: BusinessProfile) -> str:
        if not business_profile.tone_guidelines:
            return ""

        lines = "\n".join(
            f"- {guideline}" for guideline in business_profile.tone_guidelines
        )
        return f"Tone:\n{lines}"

    def _render_knowledge_sections(self, knowledge: SupportKnowledge) -> str:
        if not knowledge.sections:
            return ""

        rendered_sections: list[str] = []
        for section in knowledge.sections:
            if not section.entries:
                continue

            entries = "\n".join(f"- {entry}" for entry in section.entries)
            rendered_sections.append(f"{section.name}:\n{entries}")

        if not rendered_sections:
            return ""

        return "Business knowledge:\n" + "\n\n".join(rendered_sections)
