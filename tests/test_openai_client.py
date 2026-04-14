import unittest
from collections.abc import AsyncIterator

from openai.types.responses import ResponseTextDeltaEvent

from src.app.domain.support.models import ConversationTurn
from src.app.infrastructure.llm.openai_client import OpenAIClient


class FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeAsyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[object]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeAsyncStream(
                [
                    ResponseTextDeltaEvent(
                        content_index=0,
                        delta="Hello",
                        item_id="item_1",
                        logprobs=[],
                        output_index=0,
                        sequence_number=0,
                        type="response.output_text.delta",
                    ),
                    ResponseTextDeltaEvent(
                        content_index=0,
                        delta=" world",
                        item_id="item_1",
                        logprobs=[],
                        output_index=0,
                        sequence_number=1,
                        type="response.output_text.delta",
                    ),
                ]
            )
        return FakeResponse(output_text="Hello world")


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.responses = FakeResponsesAPI()


class OpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_uses_responses_api_and_returns_output_text(self) -> None:
        sdk_client = FakeAsyncOpenAI()
        client = OpenAIClient(api_key="test-key", model="gpt-test")
        client._client = sdk_client

        response = await client.complete(
            [ConversationTurn(role="user", content="How are you?")]
        )

        self.assertEqual(response, "Hello world")
        self.assertEqual(
            sdk_client.responses.calls,
            [
                {
                    "model": "gpt-test",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "How are you?",
                        }
                    ],
                }
            ],
        )

    async def test_stream_complete_yields_text_deltas(self) -> None:
        sdk_client = FakeAsyncOpenAI()
        client = OpenAIClient(api_key="test-key", model="gpt-test")
        client._client = sdk_client

        chunks = [
            chunk
            async for chunk in client.stream_complete(
                [ConversationTurn(role="user", content="Say hello")]
            )
        ]

        self.assertEqual(chunks, ["Hello", " world"])
        self.assertEqual(
            sdk_client.responses.calls,
            [
                {
                    "model": "gpt-test",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "Say hello",
                        }
                    ],
                    "stream": True,
                }
            ],
        )
