from dataclasses import dataclass, replace
import logging
from typing import Protocol

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

    retrieved_context: tuple[str, ...] = ()
    confidence_score: float = 0.0
    used_fallback: bool = False
    decision_reason: str = "not_evaluated"


class SupportRetrieval(Protocol):
    """Domain abstraction for retrieval decisions used by support chat."""

    def run(self, query: str) -> RetrievalDecision:
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
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker or NoOpReranker()
        self._candidate_pool_size = max(candidate_pool_size, final_top_k, 1)
        self._final_top_k = max(final_top_k, 1)

    def run(self, query: str) -> RetrievalDecision:
        try:
            results = self._retriever.retrieve(
                query=query,
                top_k=self._candidate_pool_size,
            )
        except Exception:
            logger.exception("Support retrieval failed; falling back to base prompt")
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="retrieval_error",
            )

        if not results:
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="no_results",
            )

        ranked_results = self._rerank_results(query, results)
        selected_results = ranked_results[: self._final_top_k]
        if not selected_results:
            return RetrievalDecision(
                confidence_score=0.0,
                used_fallback=True,
                decision_reason="no_results",
            )

        confidence = self._compute_confidence(selected_results)
        if confidence < CONFIDENCE_THRESHOLD:
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

    def _render_context(self, results: list[RetrievedContext]) -> tuple[str, ...]:
        rendered_context: list[str] = []
        for result in results:
            source = result.chunk.metadata.get("source")
            if source:
                rendered_context.append(f"Source: {source}\nContent: {result.chunk.text}")
            else:
                rendered_context.append(result.chunk.text)
        return tuple(rendered_context)

    def _normalize_score(self, score: float) -> float:
        return max(0.0, min(1.0, score))
