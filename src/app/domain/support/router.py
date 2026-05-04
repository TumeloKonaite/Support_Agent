from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Protocol


class RouteType(str, Enum):
    """Supported support-orchestration routes."""

    CONVERSATION = "conversation"
    RAG = "rag"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Structured support route selection."""

    route: RouteType
    reason: str
    confidence: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class SupportRouter(Protocol):
    """Abstraction for choosing the next support handling path."""

    def decide(
        self,
        user_message: str,
    ) -> RouteDecision:
        """Choose how the support request should be handled."""


class RuleBasedSupportRouter:
    """Route support requests using intent heuristics plus retrieval output."""

    _CONVERSATIONAL_PATTERNS = (
        re.compile(r"^\s*(hi|hello|hey|hiya)\b", re.IGNORECASE),
        re.compile(r"^\s*good (morning|afternoon|evening)\b", re.IGNORECASE),
        re.compile(r"^\s*(thanks|thank you|thx)\b", re.IGNORECASE),
        re.compile(r"^\s*(bye|goodbye|see you)\b", re.IGNORECASE),
        re.compile(r"\b(how are you|who are you|what can you do|what do you do)\b", re.IGNORECASE),
    )
    _INFORMATIONAL_PREFIXES = (
        "how ",
        "what ",
        "when ",
        "where ",
        "why ",
        "which ",
        "who ",
    )
    _TOOL_PATTERNS = (
        re.compile(
            r"\b(can you|could you|please|i want to|i need to|help me)\s+"
            r"(cancel|change|update|reset|refund|return|send|reschedule|book|"
            r"schedule|unsubscribe|subscribe|upgrade|downgrade|close|delete)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(cancel|change|update|reset|refund|return|send|reschedule|book|"
            r"schedule|unsubscribe|subscribe|upgrade|downgrade|close|delete)\s+"
            r"(my|the)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(track|check)\s+(my|the)\s+(order|shipment|subscription|booking)\b",
            re.IGNORECASE,
        ),
    )
    _KNOWLEDGE_PATTERNS = (
        re.compile(
            r"\b(hours|open|closing|location|address|contact|email|phone)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(refund|return|exchange|shipping|delivery|order|subscription|appointment|booking)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(policy|pricing|price|payment|billing|invoice|account|product|service|support)\b",
            re.IGNORECASE,
        ),
    )

    def decide(
        self,
        user_message: str,
    ) -> RouteDecision:
        """Choose the support route for a user request."""
        if self._is_tool_candidate(user_message):
            return RouteDecision(
                route=RouteType.TOOL,
                reason="action_request_detected",
            )

        if self._is_conversational(user_message):
            return RouteDecision(
                route=RouteType.CONVERSATION,
                reason="conversational_message_detected",
            )

        if self._needs_grounded_support_answer(user_message):
            return RouteDecision(
                route=RouteType.RAG,
                reason="knowledge_lookup_required",
            )

        return RouteDecision(
            route=RouteType.CONVERSATION,
            reason="general_assistance_response",
        )

    def _is_tool_candidate(self, user_message: str) -> bool:
        normalized_message = " ".join(user_message.split())
        if normalized_message.lower().startswith(self._INFORMATIONAL_PREFIXES):
            return False
        return any(
            pattern.search(normalized_message)
            for pattern in self._TOOL_PATTERNS
        )

    def _is_conversational(self, user_message: str) -> bool:
        normalized_message = " ".join(user_message.split())
        return any(
            pattern.search(normalized_message)
            for pattern in self._CONVERSATIONAL_PATTERNS
        )

    def _needs_grounded_support_answer(self, user_message: str) -> bool:
        normalized_message = " ".join(user_message.split())
        lowered = normalized_message.lower()
        if lowered.startswith(self._INFORMATIONAL_PREFIXES):
            return True
        return any(
            pattern.search(normalized_message)
            for pattern in self._KNOWLEDGE_PATTERNS
        )
