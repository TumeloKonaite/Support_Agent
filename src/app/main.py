from fastapi import FastAPI

from src.app.api.routes.chat import router as chat_router
from src.app.api.routes.health import router as health_router
from src.app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for basic app verification."""
    return {"message": "Support API is running"}


app.include_router(health_router)
app.include_router(chat_router)
