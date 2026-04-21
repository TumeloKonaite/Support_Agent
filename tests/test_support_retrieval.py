import json
import unittest

from src.app.domain.support.observability import SupportObservabilitySettings
from src.app.domain.support.retrieval import NoOpReranker, RetrievalPipeline
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


class ReversingReranker:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, list[RetrievedContext]]] = []

    def rerank(
        self,
        query: str,
        results: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        self.calls.append((query, list(results)))
        if self.error is not None:
            raise self.error
        return [
            RetrievedContext(
                chunk=result.chunk,
                score=1.0 - (index * 0.1),
                original_score=result.original_score,
                reranker_score=result.reranker_score,
            )
            for index, result in enumerate(reversed(results))
        ]


class RetrievalPipelineTests(unittest.TestCase):
    def _parse_log_record(self, record: str) -> dict[str, object]:
        return json.loads(record)

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

        self.assertEqual(retriever.calls, [("Can you refund this?", 10)])
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

    def test_run_logs_retrieval_scores_and_decision_metadata(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="Refunds are reviewed within two business days.", score=0.92),
                self._result(text="Escalate exceptions to the care team.", score=0.70),
            ]
        )
        pipeline = RetrievalPipeline(
            retriever=retriever,
            observability=SupportObservabilitySettings(
                prompt_preview_enabled=True,
                max_preview_chars=80,
            ),
        )

        with self.assertLogs("src.app.domain.support.retrieval", level="INFO") as logs:
            decision = pipeline.run("Refund request for order 12345", request_id="session-1")

        self.assertFalse(decision.used_fallback)
        event = self._parse_log_record(logs.records[-1].getMessage())
        self.assertEqual(event["event"], "support.retrieval.completed")
        self.assertEqual(event["request_id"], "session-1")
        self.assertEqual(event["decision_reason"], "high_confidence")
        self.assertEqual(event["used_fallback"], False)
        self.assertAlmostEqual(event["confidence"], 0.92)
        self.assertEqual(event["threshold"], 0.6)
        self.assertEqual(event["retrieved_count"], 2)
        self.assertEqual(event["selected_count"], 2)
        self.assertEqual(
            event["query"],
            {
                "length": len("Refund request for order 12345"),
                "preview": "Refund request for order [redacted-number]",
            },
        )
        first_chunk = event["retrieved_chunks"][0]
        self.assertEqual(first_chunk["rank"], 1)
        self.assertEqual(first_chunk["source"], "data/knowledge.json")
        self.assertAlmostEqual(first_chunk["score"], 0.92)
        self.assertEqual(
            first_chunk["content"]["preview"],
            "Refunds are reviewed within two business days.",
        )

    def test_run_logs_low_confidence_fallback_reason(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="General order help guidance.", score=0.62),
                self._result(text="Similar but ambiguous order guidance.", score=0.60),
            ]
        )
        pipeline = RetrievalPipeline(retriever=retriever)

        with self.assertLogs("src.app.domain.support.retrieval", level="INFO") as logs:
            decision = pipeline.run("Where is order 12345?", request_id="session-2")

        self.assertTrue(decision.used_fallback)
        event = self._parse_log_record(logs.records[-1].getMessage())
        self.assertEqual(event["decision_reason"], "low_confidence")
        self.assertEqual(event["fallback_reason"], "low_confidence")
        self.assertEqual(event["used_fallback"], True)
        self.assertAlmostEqual(event["confidence"], 0.52)
        self.assertIsNone(event["query"]["preview"])

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

    def test_run_uses_reranked_order_for_confidence_and_context(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="Weak match", score=0.61),
                self._result(text="Best match after rerank", score=0.55),
                self._result(text="Third match", score=0.40),
            ]
        )
        reranker = ReversingReranker()
        pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=reranker,
            candidate_pool_size=5,
            final_top_k=2,
        )

        decision = pipeline.run("Can you help with refunds?")

        self.assertEqual(retriever.calls, [("Can you help with refunds?", 5)])
        self.assertEqual(len(reranker.calls), 1)
        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "high_confidence")
        self.assertAlmostEqual(decision.confidence_score, 1.0)
        self.assertEqual(
            decision.retrieved_context,
            (
                "Source: data/knowledge.json\nContent: Third match",
                "Source: data/knowledge.json\nContent: Best match after rerank",
            ),
        )

    def test_run_falls_back_to_original_order_when_reranking_fails(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="Original best match", score=0.91),
                self._result(text="Original second match", score=0.50),
            ]
        )
        pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=ReversingReranker(error=RuntimeError("rerank failed")),
        )

        decision = pipeline.run("Need policy help")

        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "high_confidence")
        self.assertAlmostEqual(decision.confidence_score, 0.91)
        self.assertEqual(
            decision.retrieved_context,
            (
                "Source: data/knowledge.json\nContent: Original best match",
                "Source: data/knowledge.json\nContent: Original second match",
            ),
        )

    def test_noop_reranker_preserves_existing_behavior(self) -> None:
        retriever = RecordingRetriever(
            results=[
                self._result(text="First result", score=0.88),
                self._result(text="Second result", score=0.72),
            ]
        )
        pipeline = RetrievalPipeline(
            retriever=retriever,
            reranker=NoOpReranker(),
            final_top_k=2,
        )

        decision = pipeline.run("When will my refund arrive?")

        self.assertFalse(decision.used_fallback)
        self.assertEqual(decision.decision_reason, "high_confidence")
        self.assertAlmostEqual(decision.confidence_score, 0.88)
        self.assertEqual(
            decision.retrieved_context,
            (
                "Source: data/knowledge.json\nContent: First result",
                "Source: data/knowledge.json\nContent: Second result",
            ),
        )
