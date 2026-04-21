from dataclasses import dataclass

from src.app.domain.support.models import (
    SupportAnswer,
    SupportCitation,
    SupportContextChunk,
)
from src.app.domain.support.retrieval import RetrievalDecision


SAFE_FALLBACK_MESSAGE = (
    "I don't have enough verified information in our support knowledge to answer that "
    "confidently. Please rephrase your question or contact the human support team for help."
)


@dataclass(frozen=True, slots=True)
class GuardrailDecision:
    """Resolved answer-time guardrail behavior for a support request."""

    answer: SupportAnswer
    context_chunks: tuple[SupportContextChunk, ...] = ()
    confidence_score: float = 0.0
    retrieval_reason: str = "not_evaluated"

    @property
    def should_fallback(self) -> bool:
        return self.answer.grounding_status == "fallback"


class SupportGuardrailPolicy:
    """Apply support answer guardrails to retrieval output."""

    def evaluate(self, retrieval_decision: RetrievalDecision) -> GuardrailDecision:
        citations = self._build_citations(retrieval_decision.retrieved_context)
        if retrieval_decision.used_fallback or not retrieval_decision.retrieved_context:
            return GuardrailDecision(
                answer=SupportAnswer(
                    message=SAFE_FALLBACK_MESSAGE,
                    grounding_status="fallback",
                    fallback_reason=retrieval_decision.decision_reason,
                ),
                confidence_score=retrieval_decision.confidence_score,
                retrieval_reason=retrieval_decision.decision_reason,
            )

        return GuardrailDecision(
            answer=SupportAnswer(
                message="",
                citations=citations,
                context_chunks=retrieval_decision.retrieved_context,
                used_context=True,
                grounding_status="grounded",
            ),
            context_chunks=retrieval_decision.retrieved_context,
            confidence_score=retrieval_decision.confidence_score,
            retrieval_reason=retrieval_decision.decision_reason,
        )

    def _build_citations(
        self,
        context_chunks: tuple[SupportContextChunk, ...],
    ) -> tuple[SupportCitation, ...]:
        return tuple(
            SupportCitation(
                chunk_id=chunk.chunk_id,
                label=chunk.label,
                source=chunk.source,
            )
            for chunk in context_chunks
        )
