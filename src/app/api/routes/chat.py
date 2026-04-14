from fastapi import APIRouter

from src.app.api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    """
    Minimal placeholder chat endpoint.

    This is intentionally not wired into the existing chat internals yet.
    It only exists so the new package layout is functional.
    """
    return ChatResponse(
        response=f"Chat endpoint is wired. Received message: {payload.message}"
    )
