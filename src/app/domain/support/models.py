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
