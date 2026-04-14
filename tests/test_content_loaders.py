import json
import tempfile
import unittest
from pathlib import Path

from src.app.infrastructure.content.business_profile_loader import (
    BusinessProfileLoader,
)
from src.app.infrastructure.content.knowledge_loader import KnowledgeLoader


class ContentLoaderTests(unittest.TestCase):
    def test_business_profile_loader_reads_default_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._write_json(
                data_dir / "business_profile.json",
                {
                    "business_name": "Glow Studio",
                    "assistant_identity": "the Glow Studio support assistant",
                    "support_email": "support@glow.example",
                    "tone_guidelines": ["Be kind."],
                    "metadata": {"region": "US"},
                },
            )

            profile = BusinessProfileLoader(data_dir).load()

        self.assertEqual(profile.business_name, "Glow Studio")
        self.assertEqual(profile.support_email, "support@glow.example")
        self.assertEqual(profile.tone_guidelines, ("Be kind.",))
        self.assertEqual(profile.metadata, {"region": "US"})

    def test_knowledge_loader_prefers_tenant_specific_dataset_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._write_json(
                data_dir / "knowledge.json",
                {"sections": {"Policies": ["Use fallback policy."]}},
            )
            self._write_json(
                data_dir / "tenant-a" / "knowledge.json",
                {"sections": {"Policies": ["Use tenant policy."]}},
            )

            knowledge = KnowledgeLoader(data_dir).load(tenant_id="tenant-a")

        self.assertEqual(len(knowledge.sections), 1)
        self.assertEqual(knowledge.sections[0].entries, ("Use tenant policy.",))

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
