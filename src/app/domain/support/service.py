from collections.abc import AsyncIterator
from uuid import uuid4

from src.app.api.schemas.chat import ChatRequest, ChatResponse
from src.app.domain.support.models import ChatResult, ChatSession, ConversationTurn
from src.app.infrastructure.storage.conversation_store import ConversationStore


class SupportService:
    """Application service responsible for the chat request lifecycle."""

    def __init__(self, conversation_store: ConversationStore) -> None:
        self._conversation_store = conversation_store

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return the assistant response."""
        result = await self._run_chat_flow(request)
        return ChatResponse(response=result.response)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Process a chat request and yield the assistant response as a stream."""
        result = await self._run_chat_flow(request)
        yield result.response

    async def _run_chat_flow(self, request: ChatRequest) -> ChatResult:
        """Execute the end-to-end chat flow for a single request."""
        session = await self._load_session(request.session_id)
        messages = self._build_messages(session, request.message)
        response = await self._generate_response(messages)
        await self._persist_conversation(
            session.session_id,
            session.history,
            request.message,
            response,
        )
        return ChatResult(session_id=session.session_id, response=response)

    async def _load_session(self, session_id: str | None) -> ChatSession:
        """Resolve the chat session and load any stored conversation history."""
        resolved_session_id = session_id or str(uuid4())
        history = self._conversation_store.load(resolved_session_id)
        return ChatSession(session_id=resolved_session_id, history=history)

    def _build_messages(
        self,
        session: ChatSession,
        user_message: str,
    ) -> list[ConversationTurn]:
        """Build the model input from stored history and the new user message."""
        return [
            *session.history,
            ConversationTurn(role="user", content=user_message),
        ]

    async def _generate_response(
        self,
        messages: list[ConversationTurn],
    ) -> str:
        """
        Generate the assistant response.

        OpenAI is not wired yet, so this preserves the existing placeholder
        behavior while moving the orchestration into the service layer.
        """
        user_message = messages[-1].content
        return f"Chat endpoint is wired. Received message: {user_message}"

    async def _persist_conversation(
        self,
        session_id: str,
        history: list[ConversationTurn],
        user_message: str,
        assistant_response: str,
    ) -> None:
        """Persist the updated conversation."""
        messages = [
            *history,
            ConversationTurn(role="user", content=user_message),
            ConversationTurn(role="assistant", content=assistant_response),
        ]
        self._conversation_store.save(session_id, messages)
