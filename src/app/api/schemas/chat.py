from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoints."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(
        default=None,
        description="Optional conversation session identifier",
    )


class ChatResponse(BaseModel):
    """Response schema for chat endpoints."""

    response: str = Field(..., description="Assistant response")
