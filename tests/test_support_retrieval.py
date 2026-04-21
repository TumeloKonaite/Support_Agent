import unittest

from src.app.domain.support.retrieval import RetrievalPipeline
from src.app.infrastructure.retrieval.retriever import (
    KnowledgeChunk,
    RetrievedContext,
)


class RecordingRetriever:
    def __init__(
        self,
        results: list[RetrievedContext] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        self.calls.append((query, top_k))
        if self.error is not None:
            raise self.error
        return list(self.results)


class RetrievalPipelineTests(unittest.TestCase):
    def _result(
        self,
        *,
        text: str,
        score: float,
        source: str = "data/knowledge.json",
    ) -> RetrievedContext:
        return RetrievedContext(
            chunk=KnowledgeChunk(
                chunk_id=f"chunk-{abs(hash((text, score, source)))}",
                text=text,
                metadata={"source": source} if source else {},
            ),
            score=score,
        )

    def test_run_returns_grounded_context_for_high_confidence_results(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="Refunds are reviewed within two business days.", score=0.92),
                self._result(text="Escalate exceptions to the care team.", score=0.70),
            ]
        )
        pipeline = RetrievalPipeline(retriever=retriever)

        decision = pipeline.run("Can you refund this?")

        self.assertEqual(retriever.calls, [("Can you refund this?", None)])
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "high_confidence")
        self.assertAlmostEqual(decision.confidence_score, 0.92)
        self.assertEqual(
            decision.retrieved_context,
            (
                "Source: data/knowledge.json\nContent: Refunds are reviewed within two business days.",
                "Source: data/knowledge.json\nContent: Escalate exceptions to the care team.",
            ),
        )

    def test_run_uses_fallback_for_low_confidence_results(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="General order help guidance.", score=0.62),
                self._result(text="Similar but ambiguous order guidance.", score=0.60),
            ]
        )
        pipeline = RetrievalPipeline(retriever=retriever)

        decision = pipeline.run("Where is my order?")

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "low_confidence")
        self.assertAlmostEqual(decision.confidence_score, 0.52)
        self.assertEqual(decision.retrieved_context, ())

    def test_run_uses_fallback_when_no_results_are_found(self) -> None:
        pipeline = RetrievalPipeline(retriever=RecordingRetriever(results=[]))

        decision = pipeline.run("What are your hours?")

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "no_results")
        self.assertEqual(decision.confidence_score, 0.0)
        self.assertEqual(decision.retrieved_context, ())

    def test_run_uses_fallback_when_retrieval_raises(self) -> None:
        pipeline = RetrievalPipeline(
            retriever=RecordingRetriever(error=RuntimeError("retrieval failed"))
        )

        decision = pipeline.run("Need help with my account")

        self.assertTrue(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "retrieval_error")
        self.assertEqual(decision.confidence_score, 0.0)
        self.assertEqual(decision.retrieved_context, ())
