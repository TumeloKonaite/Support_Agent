import json
from pathlib import Path

from src.app.domain.support.models import KnowledgeSection, SupportKnowledge


class KnowledgeLoader:
    """Load support knowledge content from runtime-managed files."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(self, tenant_id: str | None = None) -> SupportKnowledge:
        """Load support knowledge for the selected tenant dataset."""
        payload = json.loads(self._resolve_path(tenant_id).read_text(encoding="utf-8"))
        sections_payload = payload.get("sections", {})
        if not isinstance(sections_payload, dict):
            raise ValueError("knowledge sections must be a JSON object")

        return SupportKnowledge(
            sections=tuple(
                KnowledgeSection(
                    name=str(section_name),
                    entries=self._string_tuple(entries),
                )
                for section_name, entries in sections_payload.items()
            )
        )

    def _resolve_path(self, tenant_id: str | None) -> Path:
        if tenant_id:
            candidate = self._data_dir / tenant_id / "knowledge.json"
            if candidate.exists():
                return candidate

        return self._data_dir / "knowledge.json"

    def _string_tuple(self, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError("knowledge section entries must be JSON arrays")
        return tuple(str(item) for item in value)
