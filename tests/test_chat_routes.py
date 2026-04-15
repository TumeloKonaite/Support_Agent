import unittest

from fastapi.testclient import TestClient

from src.app.api.schemas.chat import ChatRequest, ChatResponse
from src.app.core.dependencies import get_support_service
from src.app.main import app


class FakeSupportService:
    def __init__(self) -> None:
        self.chat_requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self.chat_requests.append(request)
        return ChatResponse(response=f"handled: {request.message}")


class ChatRouteTests(unittest.TestCase):
    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_chat_endpoint_uses_overridden_support_service(self) -> None:
        service = FakeSupportService()
        app.dependency_overrides[get_support_service] = lambda: service

        response = TestClient(app).post(
            "/chat",
            json={"message": "hello", "session_id": "session-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": "handled: hello"})
        self.assertEqual(
            service.chat_requests,
            [ChatRequest(message="hello", session_id="session-1")],
        )
