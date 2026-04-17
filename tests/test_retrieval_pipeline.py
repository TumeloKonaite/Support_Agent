import json
import tempfile
import unittest
from pathlib import Path

from src.app.core.config import Settings
from src.app.infrastructure.retrieval.indexer import (
    KnowledgeDocument,
    build_indexer,
    run_query,
)
from src.app.infrastructure.retrieval.vector_store import JsonVectorStore


class RetrievalPipelineTests(unittest.TestCase):
    def test_chunk_documents_applies_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            vector_store_path = data_dir / "vector_store.json"
            indexer = build_indexer(
                Settings(
                    content_data_dir=data_dir,
                    retrieval_vector_store_path=vector_store_path,
                    retrieval_chunk_size=30,
                    retrieval_chunk_overlap=10,
                )
            )

            chunks = indexer.chunk_documents(
                [
                    KnowledgeDocument(
                        source="memory",
                        content="abcdefghijklmnopqrstuvwxyz1234567890",
                        metadata={"kind": "test"},
                    )
                ]
            )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "abcdefghijklmnopqrstuvwxyz1234")
        self.assertEqual(chunks[1].text, "uvwxyz1234567890")

    def test_indexer_loads_json_knowledge_and_persists_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._write_json(
                data_dir / "knowledge.json",
                {
                    "sections": {
                        "Policies": [
                            "Escalate refund exceptions to a human specialist.",
                        ]
                    }
                },
            )
            vector_store_path = data_dir / "retrieval" / "vectors.json"
            indexer = build_indexer(
                Settings(
                    content_data_dir=data_dir,
                    retrieval_vector_store_path=vector_store_path,
                    retrieval_chunk_size=80,
                    retrieval_chunk_overlap=20,
                )
            )

            chunks = indexer.index()
            store = JsonVectorStore(vector_store_path)
            persisted_count = store.count()
            exists = vector_store_path.exists()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(persisted_count, 1)
        self.assertTrue(exists)

    def test_retrieval_returns_relevant_business_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            self._write_json(
                data_dir / "business_profile.json",
                {
                    "business_name": "Glow Studio",
                    "support_hours": "Monday to Friday, 9:00 AM to 5:00 PM local time.",
                },
            )
            self._write_json(
                data_dir / "knowledge.json",
                {
                    "sections": {
                        "Policies": [
                            "Escalate refund requests to the human support team.",
                            "Ask a clarifying question if the request is ambiguous.",
                        ],
                        "FAQs": [
                            "Customers often ask about products and support hours.",
                        ],
                    }
                },
            )
            settings = Settings(
                content_data_dir=data_dir,
                retrieval_vector_store_path=data_dir / "retrieval" / "vectors.json",
                retrieval_top_k=2,
                retrieval_chunk_size=120,
                retrieval_chunk_overlap=20,
            )
            indexer = build_indexer(settings)
            indexer.index()

            results = run_query(indexer, "When are your support hours?", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertIn("support hours", results[0].chunk.text.lower())

    def test_indexer_can_query_repo_data_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        data_dir = repo_root / "data"
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                content_data_dir=data_dir,
                retrieval_vector_store_path=Path(temp_dir) / "vectors.json",
                retrieval_top_k=2,
                retrieval_chunk_size=120,
                retrieval_chunk_overlap=20,
            )
            indexer = build_indexer(settings)
            indexer.index()

            results = run_query(indexer, "What are your support hours?", top_k=5)

        self.assertTrue(results)
        self.assertTrue(
            any("support hours" in result.chunk.text.lower() for result in results),
        )

    def _write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
