import json
import unittest
from collections.abc import AsyncIterator

from src.app.api.schemas.chat import ChatRequest
from src.app.domain.support.models import (
    BusinessProfile,
    ConversationTurn,
    KnowledgeSection,
    SupportKnowledge,
)
from src.app.domain.support.prompt_builder import (
    BusinessProfileSource,
    KnowledgeSource,
    SupportPromptBuilder,
)
from src.app.domain.support.retrieval import RetrievalPipeline
from src.app.domain.support.service import SupportService
from src.app.infrastructure.retrieval.retriever import (
    KnowledgeChunk,
    RetrievedContext,
)
from src.app.infrastructure.storage.conversation_store import ConversationStore


class InMemoryConversationStore(ConversationStore):
    def __init__(self, data: dict[str, list[ConversationTurn]] | None = None) -> None:
        self.data = data or {}
        self.load_calls: list[str] = []
        self.save_calls: list[tuple[str, list[ConversationTurn]]] = []

    def load(self, session_id: str) -> list[ConversationTurn]:
        self.load_calls.append(session_id)
        return list(self.data.get(session_id, []))

    def save(self, session_id: str, messages: list[ConversationTurn]) -> None:
        self.data[session_id] = list(messages)
        self.save_calls.append((session_id, list(messages)))


class RecordingOpenAIClient:
    def __init__(
        self,
        response: str = "default response",
        stream_chunks: list[str] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.response = response
        self.stream_chunks = stream_chunks
        self.stream_error = stream_error
        self.complete_calls: list[list[ConversationTurn]] = []
        self.stream_calls: list[list[ConversationTurn]] = []

    async def complete(self, messages: list[ConversationTurn]) -> str:
        self.complete_calls.append(list(messages))
        return self.response

    async def stream_complete(
        self,
        messages: list[ConversationTurn],
    ) -> AsyncIterator[str]:
        self.stream_calls.append(list(messages))
        for chunk in self.stream_chunks or [self.response]:
            yield chunk

        if self.stream_error is not None:
            raise self.stream_error


class StaticBusinessProfileSource(BusinessProfileSource):
    def __init__(self, profile: BusinessProfile) -> None:
        self._profile = profile

    def load(self, tenant_id: str | None = None) -> BusinessProfile:
        return self._profile


class StaticKnowledgeSource(KnowledgeSource):
    def __init__(self, knowledge: SupportKnowledge) -> None:
        self._knowledge = knowledge

    def load(self, tenant_id: str | None = None) -> SupportKnowledge:
        return self._knowledge


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


class SupportServiceTests(unittest.IsolatedAsyncioTestCase):
    def _parse_log_record(self, record: str) -> dict[str, object]:
        return json.loads(record)

    def _build_prompt_builder(self) -> SupportPromptBuilder:
        return SupportPromptBuilder(
            business_profile_source=StaticBusinessProfileSource(
                BusinessProfile(
                    business_name="Glow Studio",
                    assistant_identity="the Glow Studio support assistant",
                    escalation_target="Escalate refunds to the human care team.",
                    tone_guidelines=("Be warm and practical.",),
                )
            ),
            knowledge_source=StaticKnowledgeSource(
                SupportKnowledge(
                    sections=(
                        KnowledgeSection(
                            name="Policies",
                            entries=("Escalate account access issues.",),
                        ),
                    )
                )
            )
        )

    def _build_retrieved_context(self, text: str) -> RetrievedContext:
        return RetrievedContext(
            chunk=KnowledgeChunk(
                chunk_id="chunk-1",
                text=text,
                metadata={"source": "data/knowledge.json"},
            ),
            score=0.9,
        )

    def _build_retrieval_pipeline(
        self,
        retriever: RecordingRetriever | None = None,
    ) -> RetrievalPipeline:
        return RetrievalPipeline(retriever=retriever or RecordingRetriever())

    async def test_chat_uses_existing_history_and_persists_new_turns(self) -> None:
        store = InMemoryConversationStore(
            {
                "session-1": [
                    ConversationTurn(role="user", content="older question"),
                    ConversationTurn(role="assistant", content="older answer"),
                ]
            }
        )
        client = RecordingOpenAIClient(response="assistant reply")
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        response = await service.chat(
            ChatRequest(message="new question", session_id="session-1")
        )

        self.assertEqual(response.response, "I don't have enough verified information in our support knowledge to answer that confidently. Please rephrase your question or contact the human support team for help.")
        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(response.fallback_reason, "no_results")
        self.assertEqual(len(client.complete_calls), 0)
        self.assertEqual(
            store.data["session-1"][-1],
            ConversationTurn(
                role="assistant",
                content=response.response,
            ),
        )

    async def test_chat_creates_new_session_history_when_missing(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        response = await service.chat(ChatRequest(message="hello", session_id="session-2"))

        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(len(client.complete_calls), 0)
        self.assertEqual(
            store.data["session-2"],
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(role="assistant", content=response.response),
            ],
        )

    async def test_chat_generates_session_id_when_not_supplied(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        response = await service.chat(ChatRequest(message="hello"))

        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(len(store.load_calls), 1)
        generated_session_id = store.load_calls[0]
        self.assertTrue(generated_session_id)
        self.assertEqual(set(store.data), {generated_session_id})
        self.assertEqual(
            store.save_calls,
            [
                (
                    generated_session_id,
                    [
                        ConversationTurn(role="user", content="hello"),
                        ConversationTurn(
                            role="assistant",
                            content=response.response,
                        ),
                    ],
                )
            ],
        )

    async def test_stream_chat_uses_streaming_client_and_persists_response(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(stream_chunks=["stream ", "reply"])
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                ChatRequest(message="stream me", session_id="session-3")
            )
        ]

        self.assertEqual(len(client.stream_calls), 0)
        self.assertEqual(
            chunks,
            [
                "I don't have enough verified information in our support knowledge to answer that confidently. Please rephrase your question or contact the human support team for help."
            ],
        )
        self.assertEqual(
            store.data["session-3"],
            [
                ConversationTurn(role="user", content="stream me"),
                ConversationTurn(role="assistant", content=chunks[0]),
            ],
        )

    async def test_stream_chat_preserves_partial_chunks_until_completion(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(stream_chunks=["part", "ial", " response"])
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                ChatRequest(message="split it", session_id="session-4")
            )
        ]

        self.assertEqual(
            chunks,
            [
                "I don't have enough verified information in our support knowledge to answer that confidently. Please rephrase your question or contact the human support team for help."
            ],
        )
        self.assertEqual(
            store.data["session-4"],
            [
                ConversationTurn(role="user", content="split it"),
                ConversationTurn(role="assistant", content=chunks[0]),
            ],
        )

    async def test_stream_chat_propagates_errors_without_persisting_partial_output(
        self,
    ) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(
            stream_chunks=["partial"],
            stream_error=RuntimeError("stream failed"),
        )
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(),
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                ChatRequest(message="fail please", session_id="session-5")
            )
        ]

        self.assertEqual(len(client.stream_calls), 0)
        self.assertEqual(len(chunks), 1)
        self.assertIn("don't have enough verified information", chunks[0])
        self.assertIn("session-5", store.data)

    async def test_chat_calls_retriever_before_llm_and_includes_context(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        retriever = RecordingRetriever(
            results=[
                self._build_retrieved_context(
                    "Refund exceptions must be escalated to a human specialist."
                )
            ]
        )
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(retriever),
        )

        response = await service.chat(
            ChatRequest(message="Can you refund this?", session_id="session-6")
        )

        self.assertEqual(retriever.calls, [("Can you refund this?", 10)])
        self.assertIn("Retrieved business context:", client.complete_calls[0][1].content)
        self.assertIn("[1] Source: data/knowledge.json", client.complete_calls[0][1].content)
        self.assertIn(
            "Refund exceptions must be escalated to a human specialist.",
            client.complete_calls[0][1].content,
        )
        self.assertEqual(response.citations[0].label, "[1]")
        self.assertTrue(response.used_context)
        self.assertEqual(response.grounding_status, "grounded")
        self.assertTrue(response.response.endswith("Sources: [1]"))

    async def test_stream_chat_uses_retrieved_context(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(stream_chunks=["stream ", "reply"])
        retriever = RecordingRetriever(
            results=[self._build_retrieved_context("Support hours are weekdays 9 to 5.")]
        )
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(retriever),
        )

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                ChatRequest(message="When are you open?", session_id="session-7")
            )
        ]

        self.assertEqual(chunks, ["stream ", "reply", "\n\nSources: [1]"])
        self.assertEqual(retriever.calls, [("When are you open?", 10)])
        self.assertIn("Retrieved business context:", client.stream_calls[0][1].content)
        self.assertIn("Support hours are weekdays 9 to 5.", client.stream_calls[0][1].content)

    async def test_chat_handles_empty_retrieval_without_prompt_section(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(
                RecordingRetriever(results=[])
            ),
        )

        response = await service.chat(ChatRequest(message="hello", session_id="session-8"))

        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(response.fallback_reason, "no_results")
        self.assertEqual(len(client.complete_calls), 0)

    async def test_chat_handles_retrieval_failures_without_crashing(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(
                RecordingRetriever(error=RuntimeError("retrieval failed"))
            ),
        )

        response = await service.chat(
            ChatRequest(message="Need help with my order", session_id="session-9")
        )

        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(response.fallback_reason, "retrieval_error")
        self.assertEqual(len(client.complete_calls), 0)

    async def test_chat_omits_low_confidence_retrieval_context(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="assistant reply")
        retriever = RecordingRetriever(
            results=[
                RetrievedContext(
                    chunk=KnowledgeChunk(
                        chunk_id="chunk-1",
                        text="Standard returns are processed within five business days.",
                        metadata={"source": "data/knowledge.json"},
                    ),
                    score=0.62,
                ),
                RetrievedContext(
                    chunk=KnowledgeChunk(
                        chunk_id="chunk-2",
                        text="Refund approval timelines vary by payment method.",
                        metadata={"source": "data/knowledge.json"},
                    ),
                    score=0.60,
                ),
            ]
        )
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(retriever),
        )

        response = await service.chat(
            ChatRequest(message="How long do refunds take?", session_id="session-10")
        )

        self.assertEqual(response.grounding_status, "fallback")
        self.assertEqual(retriever.calls, [("How long do refunds take?", 10)])
        self.assertEqual(len(client.complete_calls), 0)

    async def test_chat_logs_model_input_summary(self) -> None:
        store = InMemoryConversationStore(
            {
                "session-11": [
                    ConversationTurn(role="user", content="older question"),
                    ConversationTurn(role="assistant", content="older answer"),
                ]
            }
        )
        client = RecordingOpenAIClient(response="assistant reply")
        retriever = RecordingRetriever(
            results=[self._build_retrieved_context("Refunds are reviewed within two days.")]
        )
        service = SupportService(
            conversation_store=store,
            openai_client=client,
            prompt_builder=self._build_prompt_builder(),
            retrieval_pipeline=self._build_retrieval_pipeline(retriever),
        )

        with self.assertLogs("src.app.domain.support.service", level="INFO") as logs:
            await service.chat(
                ChatRequest(message="Can you refund order 12345?", session_id="session-11")
            )

        event = self._parse_log_record(logs.records[-1].getMessage())
        self.assertEqual(event["event"], "support.request.model_input")
        self.assertEqual(event["request_id"], "session-11")
        self.assertEqual(event["history_turn_count"], 2)
        self.assertEqual(event["retrieval_used_fallback"], False)
        self.assertAlmostEqual(event["retrieval_confidence"], 0.9)
        self.assertEqual(event["retrieval_decision_reason"], "high_confidence")
        self.assertEqual(event["retrieved_context_count"], 1)
        self.assertEqual(event["guardrail_status"], "grounded")
        self.assertIsNone(event["guardrail_fallback_reason"])
