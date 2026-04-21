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

        self.assertEqual(response.response, "assistant reply")
        self.assertEqual(len(client.complete_calls), 1)
        sent_messages = client.complete_calls[0]
        self.assertEqual(sent_messages[0].role, "system")
        self.assertIn("Glow Studio", sent_messages[0].content)
        self.assertEqual(sent_messages[1].role, "user")
        self.assertIn("User: older question", sent_messages[1].content)
        self.assertIn("Assistant: older answer", sent_messages[1].content)
        self.assertIn("Latest customer message:\nnew question", sent_messages[1].content)
        self.assertEqual(
            store.data["session-1"],
            [
                ConversationTurn(role="user", content="older question"),
                ConversationTurn(role="assistant", content="older answer"),
                ConversationTurn(role="user", content="new question"),
                ConversationTurn(role="assistant", content="assistant reply"),
            ],
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

        self.assertEqual(response.response, "assistant reply")
        self.assertIn(
            "No previous conversation history.",
            client.complete_calls[0][1].content,
        )
        self.assertEqual(
            store.data["session-2"],
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(role="assistant", content="assistant reply"),
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

        self.assertEqual(response.response, "assistant reply")
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
                            content="assistant reply",
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

        self.assertEqual(
            chunks,
            ["stream ", "reply"],
        )
        self.assertEqual(len(client.stream_calls), 1)
        sent_messages = client.stream_calls[0]
        self.assertEqual(sent_messages[0].role, "system")
        self.assertEqual(sent_messages[1].role, "user")
        self.assertIn("Latest customer message:\nstream me", sent_messages[1].content)
        self.assertEqual(
            store.data["session-3"],
            [
                ConversationTurn(role="user", content="stream me"),
                ConversationTurn(role="assistant", content="stream reply"),
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

        self.assertEqual(chunks, ["part", "ial", " response"])
        self.assertEqual(
            store.data["session-4"],
            [
                ConversationTurn(role="user", content="split it"),
                ConversationTurn(role="assistant", content="partial response"),
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

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            [
                chunk
                async for chunk in service.stream_chat(
                    ChatRequest(message="fail please", session_id="session-5")
                )
            ]

        self.assertNotIn("session-5", store.data)

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

        await service.chat(ChatRequest(message="Can you refund this?", session_id="session-6"))

        self.assertEqual(retriever.calls, [("Can you refund this?", None)])
        self.assertIn("Retrieved business context:", client.complete_calls[0][1].content)
        self.assertIn(
            "Refund exceptions must be escalated to a human specialist.",
            client.complete_calls[0][1].content,
        )

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

        self.assertEqual(chunks, ["stream ", "reply"])
        self.assertEqual(retriever.calls, [("When are you open?", None)])
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

        await service.chat(ChatRequest(message="hello", session_id="session-8"))

        self.assertNotIn("Retrieved business context:", client.complete_calls[0][1].content)

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

        self.assertEqual(response.response, "assistant reply")
        self.assertNotIn("Retrieved business context:", client.complete_calls[0][1].content)

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

        self.assertEqual(response.response, "assistant reply")
        self.assertEqual(retriever.calls, [("How long do refunds take?", None)])
        self.assertNotIn("Retrieved business context:", client.complete_calls[0][1].content)
