from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")


@dataclass(frozen=True, slots=True)
class SupportObservabilitySettings:
    """Controls how much support-flow debug data is emitted to logs."""

    enabled: bool = True
    prompt_preview_enabled: bool = False
    redact_sensitive_fields: bool = True
    max_preview_chars: int = 160


def log_support_event(
    logger: logging.Logger,
    *,
    event: str,
    payload: dict[str, Any],
    level: int = logging.INFO,
) -> None:
    """Emit a machine-readable support observability event."""
    logger.log(
        level,
        json.dumps(
            {
                "event": event,
                **payload,
            },
            sort_keys=True,
            default=_json_default,
        ),
    )


def preview_text(
    text: str,
    settings: SupportObservabilitySettings,
) -> str | None:
    """Return a bounded preview of text when previews are enabled."""
    if not settings.prompt_preview_enabled:
        return None

    normalized = " ".join(text.split())
    if settings.redact_sensitive_fields:
        normalized = redact_text(normalized)

    if len(normalized) <= settings.max_preview_chars:
        return normalized
    return normalized[: settings.max_preview_chars - 3].rstrip() + "..."


def summarize_text(
    text: str,
    settings: SupportObservabilitySettings,
) -> dict[str, Any]:
    """Return safe summary metadata for potentially sensitive content."""
    normalized = " ".join(text.split())
    return {
        "length": len(text),
        "preview": preview_text(normalized, settings),
    }


def redact_text(text: str) -> str:
    """Apply lightweight redaction to common sensitive patterns."""
    redacted = EMAIL_RE.sub("[redacted-email]", text)
    return LONG_NUMBER_RE.sub("[redacted-number]", redacted)

def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
