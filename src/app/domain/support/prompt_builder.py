from src.app.domain.support.models import (
    ConversationTurn,
    PromptBuildInput,
    PromptBuildResult,
)
from src.app.domain.support.policies import (
    DEFAULT_SUPPORT_POLICIES,
    SupportPolicies,
)


class SupportPromptBuilder:
    """Build support-domain prompts for the LLM layer."""

    def __init__(self, policies: SupportPolicies = DEFAULT_SUPPORT_POLICIES) -> None:
        self._policies = policies

    def build(self, prompt_input: PromptBuildInput) -> PromptBuildResult:
        """Generate the system and user prompts for a support interaction."""
        return PromptBuildResult(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(
                history=prompt_input.history,
                user_message=prompt_input.user_message,
            ),
        )

    def _build_system_prompt(self) -> str:
        tone_section = "\n".join(
            f"- {guideline}" for guideline in self._policies.tone_guidelines
        )
        rules_section = "\n".join(
            f"- {rule}" for rule in self._policies.operating_rules
        )
        constraints_section = "\n".join(
            f"- {constraint}" for constraint in self._policies.constraints
        )
        return (
            f"{self._policies.assistant_role}\n\n"
            "Tone:\n"
            f"{tone_section}\n\n"
            "Operating rules:\n"
            f"{rules_section}\n\n"
            "Constraints:\n"
            f"{constraints_section}"
        )

    def _build_user_prompt(
        self,
        history: list[ConversationTurn],
        user_message: str,
    ) -> str:
        history_text = self._render_history(history)
        return (
            "Support conversation context:\n"
            f"{history_text}\n\n"
            "Latest customer message:\n"
            f"{user_message}\n\n"
            "Respond as the support assistant."
        )

    def _render_history(self, history: list[ConversationTurn]) -> str:
        if not history:
            return "No previous conversation history."

        return "\n".join(
            f"{turn.role.title()}: {turn.content}"
            for turn in history
        )
