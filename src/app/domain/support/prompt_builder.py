import logging
from typing import Protocol

from src.app.domain.support.models import (
    BusinessProfile,
    ConversationTurn,
    PromptBuildInput,
    PromptBuildResult,
    SupportKnowledge,
)
from src.app.domain.support.observability import (
    SupportObservabilitySettings,
    log_support_event,
    summarize_text,
)


logger = logging.getLogger(__name__)


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
        observability: SupportObservabilitySettings | None = None,
    ) -> None:
        self._business_profile_source = business_profile_source
        self._knowledge_source = knowledge_source
        self._observability = observability or SupportObservabilitySettings()

    def build(self, prompt_input: PromptBuildInput) -> PromptBuildResult:
        """Generate the system and user prompts for a support interaction."""
        business_profile = self._business_profile_source.load(prompt_input.tenant_id)
        knowledge = self._knowledge_source.load(prompt_input.tenant_id)
        system_prompt, system_sections = self._build_system_prompt(business_profile, knowledge)
        user_prompt, user_sections = self._build_user_prompt(
            history=prompt_input.history,
            user_message=prompt_input.user_message,
            retrieved_context=prompt_input.retrieved_context,
        )
        self._log_prompt_assembly(
            prompt_input=prompt_input,
            system_sections=system_sections,
            user_sections=user_sections,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return PromptBuildResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_system_prompt(
        self,
        business_profile: BusinessProfile,
        knowledge: SupportKnowledge,
    ) -> tuple[str, list[str]]:
        section_specs = [
            ("identity", self._render_identity_section(business_profile)),
            ("contact", self._render_contact_section(business_profile)),
            ("tone", self._render_tone_section(business_profile)),
            ("knowledge", self._render_knowledge_sections(knowledge)),
        ]
        included_sections = [name for name, content in section_specs if content]
        return (
            "\n\n".join(content for _, content in section_specs if content),
            included_sections,
        )

    def _build_user_prompt(
        self,
        history: list[ConversationTurn],
        user_message: str,
        retrieved_context: tuple[str, ...],
    ) -> tuple[str, list[str]]:
        history_text = self._render_history(history)
        sections = [
            "Support conversation context:\n"
            f"{history_text}"
        ]
        included_sections = ["conversation_context"]
        retrieved_context_text = self._render_retrieved_context(retrieved_context)
        if retrieved_context_text:
            sections.append(retrieved_context_text)
            included_sections.append("retrieved_business_context")
        sections.append(
            "Latest customer message:\n"
            f"{user_message}\n\n"
            "Respond as the support assistant."
        )
        included_sections.append("latest_customer_message")
        return "\n\n".join(sections), included_sections

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

    def _log_prompt_assembly(
        self,
        *,
        prompt_input: PromptBuildInput,
        system_sections: list[str],
        user_sections: list[str],
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        if not self._observability.enabled:
            return

        log_support_event(
            logger,
            event="support.prompt.assembled",
            payload={
                "request_id": prompt_input.request_id,
                "tenant_id": prompt_input.tenant_id,
                "included_retrieval_context": bool(prompt_input.retrieved_context),
                "retrieved_chunk_count": len(prompt_input.retrieved_context),
                "retrieved_context_size_chars": sum(
                    len(item) for item in prompt_input.retrieved_context
                ),
                "history_turn_count": len(prompt_input.history),
                "system_prompt_size_chars": len(system_prompt),
                "user_prompt_size_chars": len(user_prompt),
                "system_sections": system_sections,
                "user_sections": user_sections,
                "user_message": summarize_text(
                    prompt_input.user_message,
                    self._observability,
                ),
                "retrieved_context_previews": [
                    summarize_text(item, self._observability)
                    for item in prompt_input.retrieved_context
                ],
            },
        )
