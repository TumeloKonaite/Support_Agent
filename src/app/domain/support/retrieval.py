from dataclasses import dataclass, replace
import logging
from typing import Protocol

from src.app.domain.support.models import SupportContextChunk
from src.app.domain.support.observability import (
    SupportObservabilitySettings,
    log_support_event,
    summarize_text,
)
from src.app.infrastructure.retrieval.retriever import RetrievedContext, Retriever


logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6
SPREAD_DELTA = 0.05
LOW_CONFIDENCE_PENALTY = 0.1
DEFAULT_CANDIDATE_POOL_SIZE = 10
DEFAULT_FINAL_TOP_K = 3


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    """Structured result returned by the support retrieval pipeline."""

    retrieved_context: tuple[SupportContextChunk, ...] = ()
    confidence_score: float = 0.0
    used_fallback: bool = False
    decision_reason: str = "not_evaluated"


class SupportRetrieval(Protocol):
    """Domain abstraction for retrieval decisions used by support chat."""

    def run(
        self,
        query: str,
        *,
        request_id: str | None = None,
    ) -> RetrievalDecision:
        """Retrieve and evaluate knowledge for a user query."""


class Reranker(Protocol):
    """Abstraction for reordering retrieved context by query relevance."""

    def rerank(
        self,
        query: str,
        results: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """Return retrieved results in preferred order for downstream evaluation."""


class NoOpReranker:
    """Default reranker that preserves the retriever-provided ordering."""

    def rerank(
        self,
        query: str,
        results: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        del query
        return list(results)


class RetrievalPipeline:
    """Orchestrate support retrieval and decide when grounding is trustworthy."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        observability: SupportObservabilitySettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or NoOpReranker()
        self._candidate_pool_size = max(candidate_pool_size, final_top_k, 1)
        self._final_top_k = max(final_top_k, 1)
        self._observability = observability or SupportObservabilitySettings()

    def run(
        self,
        query: str,
        *,
        request_id: str | None = None,
    ) -> RetrievalDecision:
        try:
            results = self._retriever.retrieve(
                query=query,
                top_k=self._candidate_pool_size,
            )
        except Exception:
            logger.exception("Support retrieval failed; falling back to base prompt")
            self._log_event(
                "support.retrieval.error",
                {
                    "request_id": request_id,
                    "query": summarize_text(query, self._observability),
                    "candidate_pool_size": self._candidate_pool_size,
                    "final_top_k": self._final_top_k,
                    "fallback_reason": "retrieval_error",
                },
            )
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="retrieval_error",
            )

        if not results:
            self._log_event(
                "support.retrieval.completed",
                {
                    "request_id": request_id,
                    "query": summarize_text(query, self._observability),
                    "candidate_pool_size": self._candidate_pool_size,
                    "final_top_k": self._final_top_k,
                    "retrieved_count": 0,
                    "selected_count": 0,
                    "confidence": 0.0,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "used_fallback": True,
                    "decision_reason": "no_results",
                    "fallback_reason": "no_results",
                    "retrieved_chunks": [],
                    "selected_chunks": [],
                },
            )
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="no_results",
            )

        ranked_results = self._rerank_results(query, results)
        selected_results = ranked_results[: self._final_top_k]
        if not selected_results:
            self._log_event(
                "support.retrieval.completed",
                {
                    "request_id": request_id,
                    "query": summarize_text(query, self._observability),
                    "candidate_pool_size": self._candidate_pool_size,
                    "final_top_k": self._final_top_k,
                    "retrieved_count": len(ranked_results),
                    "selected_count": 0,
                    "confidence": 0.0,
                    "threshold": CONFIDENCE_THRESHOLD,
                    "used_fallback": True,
                    "decision_reason": "no_results",
                    "fallback_reason": "no_results",
                    "retrieved_chunks": self._serialize_results(ranked_results),
                    "selected_chunks": [],
                },
            )
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="no_results",
            )

        confidence = self._compute_confidence(selected_results)
        should_fallback = confidence < CONFIDENCE_THRESHOLD
        decision_reason = "low_confidence" if should_fallback else "high_confidence"
        fallback_reason = decision_reason if should_fallback else None
        self._log_event(
            "support.retrieval.completed",
            {
                "request_id": request_id,
                "query": summarize_text(query, self._observability),
                "candidate_pool_size": self._candidate_pool_size,
                "final_top_k": self._final_top_k,
                "retrieved_count": len(ranked_results),
                "selected_count": len(selected_results),
                "confidence": confidence,
                "threshold": CONFIDENCE_THRESHOLD,
                "used_fallback": should_fallback,
                "decision_reason": decision_reason,
                "fallback_reason": fallback_reason,
                "retrieved_chunks": self._serialize_results(ranked_results),
                "selected_chunks": self._serialize_results(selected_results),
            },
        )
        if should_fallback:
            return RetrievalDecision(
                confidence_score=confidence,
                used_fallback=True,
                decision_reason="low_confidence",
            )

        return RetrievalDecision(
            retrieved_context=self._render_context(selected_results),
            confidence_score=confidence,
            used_fallback=False,
            decision_reason="high_confidence",
        )

    def _rerank_results(
        self,
        query: str,
        results: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        annotated_results = [
            replace(result, original_score=result.score)
            for result in results
        ]
        try:
            reranked_results = self._reranker.rerank(query=query, results=annotated_results)
        except Exception:
            logger.exception(
                "Support reranking failed; using original retrieval order"
            )
            return annotated_results

        return [
            replace(
                result,
                original_score=(
                    result.original_score
                    if result.original_score is not None
                    else result.score
                ),
                reranker_score=result.score,
            )
            for result in reranked_results
        ]

    def _compute_confidence(self, results: list[RetrievedContext]) -> float:
        top_score = self._normalize_score(results[0].score)
        if len(results) == 1:
            return top_score

        second_score = self._normalize_score(results[1].score)
        confidence = top_score
        if top_score - second_score < SPREAD_DELTA:
            confidence -= LOW_CONFIDENCE_PENALTY
        return self._normalize_score(confidence)

    def _render_context(
        self,
        results: list[RetrievedContext],
    ) -> tuple[SupportContextChunk, ...]:
        rendered_context: list[SupportContextChunk] = []
        for rank, result in enumerate(results, start=1):
            rendered_context.append(
                SupportContextChunk(
                    chunk_id=result.chunk.chunk_id,
                    label=f"[{rank}]",
                    text=result.chunk.text,
                    source=result.chunk.metadata.get("source"),
                    score=result.score,
                )
            )
        return tuple(rendered_context)

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))

    def _serialize_results(
        self,
        results: list[RetrievedContext],
    ) -> list[dict[str, object]]:
        serialized: list[dict[str, object]] = []
        for rank, result in enumerate(results, start=1):
            serialized.append(
                {
                    "rank": rank,
                    "chunk_id": result.chunk.chunk_id,
                    "source": result.chunk.metadata.get("source"),
                    "document_id": result.chunk.metadata.get("document_id"),
                    "score": result.score,
                    "original_score": result.original_score,
                    "reranker_score": result.reranker_score,
                    "content": summarize_text(result.chunk.text, self._observability),
                }
            )
        return serialized

    def _log_event(self, event: str, payload: dict[str, object]) -> None:
        if not self._observability.enabled:
            return
        log_support_event(logger, event=event, payload=payload)
