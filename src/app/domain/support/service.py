from collections.abc import AsyncIterator
import logging
from uuid import uuid4

from src.app.api.schemas.chat import ChatRequest, ChatResponse
from src.app.domain.support.models import (
    ChatSession,
    ConversationTurn,
    PromptBuildInput,
)
from src.app.domain.support.prompt_builder import SupportPromptBuilder
from src.app.infrastructure.llm.openai_client import LLMClient
from src.app.infrastructure.retrieval.retriever import Retriever
from src.app.infrastructure.storage.conversation_store import ConversationStore

logger = logging.getLogger(__name__)


class SupportService:
    """Application service responsible for the chat request lifecycle."""

    def __init__(
        self,
        conversation_store: ConversationStore,
        openai_client: LLMClient,
        prompt_builder: SupportPromptBuilder,
        retriever: Retriever,
    ) -> None:
        self._conversation_store = conversation_store
        self._openai_client = openai_client
        self._prompt_builder = prompt_builder
        self._retriever = retriever

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request and return the assistant response."""
        session = await self._load_session(request.session_id)
        messages = self._build_messages(session, request.message)
        response = await self._openai_client.complete(messages)
        await self._persist_conversation(
            session.session_id,
            session.history,
            request.message,
            response,
        )
        return ChatResponse(response=response)

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        """Process a chat request and yield the assistant response as a stream."""
        session = await self._load_session(request.session_id)
        messages = self._build_messages(session, request.message)
        response_chunks: list[str] = []
        async for chunk in self._openai_client.stream_complete(messages):
            response_chunks.append(chunk)
            yield chunk

        response = "".join(response_chunks)
        await self._persist_conversation(
            session.session_id,
            session.history,
            request.message,
            response,
        )

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
        retrieved_context = self._retrieve_context(user_message)
        prompt = self._prompt_builder.build(
            PromptBuildInput(
                history=session.history,
                user_message=user_message,
                retrieved_context=retrieved_context,
            )
        )
        return [
            ConversationTurn(role="system", content=prompt.system_prompt),
            ConversationTurn(role="user", content=prompt.user_prompt),
        ]

    def _retrieve_context(self, user_message: str) -> tuple[str, ...]:
        try:
            results = self._retriever.retrieve(query=user_message)
        except Exception:
            logger.exception("Support retrieval failed; continuing without retrieved context")
            return ()

        rendered_context: list[str] = []
        for result in results:
            source = result.chunk.metadata.get("source")
            if source:
                rendered_context.append(f"Source: {source}\nContent: {result.chunk.text}")
            else:
                rendered_context.append(result.chunk.text)
        return tuple(rendered_context)

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
