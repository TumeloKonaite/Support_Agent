from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Minimal request schema for chat endpoints."""

    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Minimal response schema for chat endpoints."""

    response: str = Field(..., description="Assistant response")
