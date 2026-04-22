from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Protocol

from src.app.domain.support.retrieval import RetrievalDecision


class RouteType(str, Enum):
    """Supported support-orchestration routes."""

    RAG = "rag"
    FALLBACK = "fallback"
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
        retrieval_decision: RetrievalDecision,
    ) -> RouteDecision:
        """Choose how the support request should be handled."""


class RuleBasedSupportRouter:
    """Route support requests using intent heuristics plus retrieval output."""

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

    def decide(
        self,
        user_message: str,
        retrieval_decision: RetrievalDecision,
    ) -> RouteDecision:
        """Choose the support route for a user request."""
        if self._is_tool_candidate(user_message):
            return RouteDecision(
                route=RouteType.TOOL,
                reason="action_request_detected",
                confidence=retrieval_decision.confidence_score,
            )

        if retrieval_decision.used_fallback or not retrieval_decision.retrieved_context:
            return RouteDecision(
                route=RouteType.FALLBACK,
                reason=retrieval_decision.decision_reason,
                confidence=retrieval_decision.confidence_score,
            )

        return RouteDecision(
            route=RouteType.RAG,
            reason="grounded_retrieval_available",
            confidence=retrieval_decision.confidence_score,
        )

    def _is_tool_candidate(self, user_message: str) -> bool:
        normalized_message = " ".join(user_message.split())
        if normalized_message.lower().startswith(self._INFORMATIONAL_PREFIXES):
            return False
        return any(
            pattern.search(normalized_message)
            for pattern in self._TOOL_PATTERNS
        )
