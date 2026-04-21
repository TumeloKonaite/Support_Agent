from dataclasses import dataclass, field


@dataclass(slots=True)
class ConversationTurn:
    """A single stored message in the conversation history."""

    role: str
    content: str


@dataclass(slots=True)
class ChatSession:
    """Resolved chat session state used during request processing."""

    session_id: str
    history: list[ConversationTurn] = field(default_factory=list)


@dataclass(slots=True)
class ChatResult:
    """Internal result returned by the support service."""

    session_id: str
    response: str


@dataclass(slots=True)
class PromptBuildInput:
    """Domain input required to generate a support prompt."""

    history: list[ConversationTurn] = field(default_factory=list)
    user_message: str = ""
    tenant_id: str | None = None
    request_id: str | None = None
    retrieved_context: tuple[str, ...] = ()


@dataclass(slots=True)
class PromptBuildResult:
    """Structured prompt output produced by the support prompt builder."""

    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class BusinessProfile:
    """Business identity and support-operations content used in prompts."""

    business_name: str
    assistant_identity: str
    support_email: str | None = None
    support_phone: str | None = None
    escalation_target: str | None = None
    support_hours: str | None = None
    tone_guidelines: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeSection:
    """A named section of business-managed support knowledge."""

    name: str
    entries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SupportKnowledge:
    """Structured support knowledge loaded from runtime-managed content files."""

    sections: tuple[KnowledgeSection, ...] = ()
