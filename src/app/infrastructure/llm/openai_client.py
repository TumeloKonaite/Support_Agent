from collections.abc import AsyncIterator
from typing import Protocol

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from src.app.domain.support.models import ConversationTurn


class LLMClient(Protocol):
    """Protocol for LLM providers used by the service layer."""

    async def complete(self, messages: list[ConversationTurn]) -> str:
        """Execute a non-streaming completion request."""

    async def stream_complete(
        self,
        messages: list[ConversationTurn],
    ) -> AsyncIterator[str]:
        """Execute a streaming completion request."""


class OpenAIClient:
    """Infrastructure client responsible for model execution details."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._model

    async def complete(self, messages: list[ConversationTurn]) -> str:
        """Execute a non-streaming completion request."""
        response = await self._client.responses.create(
            model=self._model,
            input=self._build_input(messages),
        )
        return response.output_text

    async def stream_complete(
        self,
        messages: list[ConversationTurn],
    ) -> AsyncIterator[str]:
        """Execute a streaming completion request."""
        stream = await self._client.responses.create(
            model=self._model,
            input=self._build_input(messages),
            stream=True,
        )
        async for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                yield event.delta

    def _build_input(self, messages: list[ConversationTurn]) -> list[dict[str, str]]:
        """Translate conversation turns into Responses API input items."""
        return [
            {
                "type": "message",
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]
