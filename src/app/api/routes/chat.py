from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.app.api.schemas.chat import ChatRequest, ChatResponse
from src.app.core.dependencies import get_support_service
from src.app.domain.support.service import SupportService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: SupportService = Depends(get_support_service),
) -> ChatResponse:
    """Handle the non-streaming chat endpoint."""
    return await service.chat(payload)


@router.post("/stream")
async def stream_chat(
    payload: ChatRequest,
    service: SupportService = Depends(get_support_service),
) -> StreamingResponse:
    """Handle the streaming chat endpoint."""
    return StreamingResponse(service.stream_chat(payload), media_type="text/plain")
