import unittest

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
        service = SupportService(conversation_store=store)

        response = await service.chat(
            ChatRequest(message="new question", session_id="session-1")
        )

        self.assertEqual(
            response.response,
            "Chat endpoint is wired. Received message: new question",
        )
        self.assertEqual(
            store.data["session-1"],
            [
                ConversationTurn(role="user", content="older question"),
                ConversationTurn(role="assistant", content="older answer"),
                ConversationTurn(role="user", content="new question"),
                ConversationTurn(
                    role="assistant",
                    content="Chat endpoint is wired. Received message: new question",
                ),
            ],
        )

    async def test_chat_creates_new_session_history_when_missing(self) -> None:
        store = InMemoryConversationStore()
        service = SupportService(conversation_store=store)

        response = await service.chat(ChatRequest(message="hello", session_id="session-2"))

        self.assertEqual(
            response.response,
            "Chat endpoint is wired. Received message: hello",
        )
        self.assertEqual(
            store.data["session-2"],
            [
                ConversationTurn(role="user", content="hello"),
                ConversationTurn(
                    role="assistant",
                    content="Chat endpoint is wired. Received message: hello",
                ),
            ],
        )
