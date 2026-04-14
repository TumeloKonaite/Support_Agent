import unittest
from collections.abc import AsyncIterator

from src.app.api.schemas.chat import ChatRequest
from src.app.domain.support.models import ConversationTurn
from src.app.domain.support.service import SupportService
from src.app.infrastructure.storage.conversation_store import ConversationStore


class InMemoryConversationStore(ConversationStore):
    def __init__(self, data: dict[str, list[ConversationTurn]] | None = None) -> None:
        self.data = data or {}

    def load(self, session_id: str) -> list[ConversationTurn]:
        return list(self.data.get(session_id, []))

    def save(self, session_id: str, messages: list[ConversationTurn]) -> None:
        self.data[session_id] = list(messages)


class RecordingOpenAIClient:
    def __init__(self, response: str = "default response") -> None:
        self.response = response
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
        yield self.response


class SupportServiceTests(unittest.IsolatedAsyncioTestCase):
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
        service = SupportService(conversation_store=store, openai_client=client)

        response = await service.chat(
            ChatRequest(message="new question", session_id="session-1")
        )

        self.assertEqual(response.response, "assistant reply")
        self.assertEqual(
            client.complete_calls,
            [
                [
                    ConversationTurn(role="user", content="older question"),
                    ConversationTurn(role="assistant", content="older answer"),
                    ConversationTurn(role="user", content="new question"),
                ]
            ],
        )
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
        service = SupportService(conversation_store=store, openai_client=client)

        response = await service.chat(ChatRequest(message="hello", session_id="session-2"))

        self.assertEqual(response.response, "assistant reply")
        self.assertEqual(
            store.data["session-2"],
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(role="assistant", content="assistant reply"),
            ],
        )

    async def test_stream_chat_uses_streaming_client_and_persists_response(self) -> None:
        store = InMemoryConversationStore()
        client = RecordingOpenAIClient(response="stream reply")
        service = SupportService(conversation_store=store, openai_client=client)

        chunks = [
            chunk
            async for chunk in service.stream_chat(
                ChatRequest(message="stream me", session_id="session-3")
            )
        ]

        self.assertEqual(
            chunks,
            ["stream reply"],
        )
        self.assertEqual(
            client.stream_calls,
            [[ConversationTurn(role="user", content="stream me")]],
        )
        self.assertEqual(
            store.data["session-3"],
            [
                ConversationTurn(role="user", content="stream me"),
                ConversationTurn(role="assistant", content="stream reply"),
            ],
        )
