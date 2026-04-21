from pydantic import BaseModel, Field


class ChatCitation(BaseModel):
    """Citation metadata returned for grounded chat responses."""

    chunk_id: str = Field(..., description="Retrieved chunk identifier")
    label: str = Field(..., description="Rendered citation label")
    source: str | None = Field(default=None, description="Optional source reference")


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
    citations: list[ChatCitation] = Field(
        default_factory=list,
        description="Citations for grounded responses",
    )
    used_context: bool = Field(
        default=False,
        description="Whether retrieved business context was used",
    )
    grounding_status: str = Field(
        default="ungrounded",
        description="Whether the answer was grounded or a fallback",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Guardrail fallback reason when grounding failed",
    )
