import json
from pathlib import Path

from src.app.domain.support.models import BusinessProfile


class BusinessProfileLoader:
    """Load structured business profile content from runtime-managed files."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(self, tenant_id: str | None = None) -> BusinessProfile:
        """Load business profile content for the selected tenant dataset."""
        payload = json.loads(self._resolve_path(tenant_id).read_text(encoding="utf-8"))
        return BusinessProfile(
            business_name=str(payload["business_name"]),
            assistant_identity=str(payload["assistant_identity"]),
            support_email=self._optional_string(payload.get("support_email")),
            support_phone=self._optional_string(payload.get("support_phone")),
            escalation_target=self._optional_string(payload.get("escalation_target")),
            support_hours=self._optional_string(payload.get("support_hours")),
            tone_guidelines=self._string_tuple(payload.get("tone_guidelines")),
            metadata=self._string_dict(payload.get("metadata")),
        )

    def _resolve_path(self, tenant_id: str | None) -> Path:
        if tenant_id:
            candidate = self._data_dir / tenant_id / "business_profile.json"
            if candidate.exists():
                return candidate

        return self._data_dir / "business_profile.json"

    def _string_tuple(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError("business profile list fields must be JSON arrays")
        return tuple(str(item) for item in value)

    def _string_dict(self, value: object) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("business profile metadata must be a JSON object")
        return {str(key): str(item) for key, item in value.items()}

    def _optional_string(self, value: object) -> str | None:
        if value is None:
            return None
        return str(value)
