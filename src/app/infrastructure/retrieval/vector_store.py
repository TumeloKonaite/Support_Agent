import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """Persistent vector-store record for a knowledge chunk."""

    chunk_id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """A vector-store search result with similarity score."""

    chunk_id: str
    text: str
    metadata: dict[str, str]
    score: float


class VectorStore(Protocol):
    """Abstraction for persistence and nearest-neighbor search."""

    def upsert(self, records: list[VectorRecord]) -> None:
        """Persist records in the vector store."""

    def search(self, query_embedding: list[float], top_k: int) -> list[VectorMatch]:
        """Return top matching records for an embedding query."""

    def clear(self) -> None:
        """Remove all stored records."""

    def count(self) -> int:
        """Return the number of persisted records."""


class JsonVectorStore:
    """Small local vector store persisted as JSON on disk."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def upsert(self, records: list[VectorRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in records]
        self._path.write_text(json.dumps(payload), encoding="utf-8")

    def search(self, query_embedding: list[float], top_k: int) -> list[VectorMatch]:
        matches = [
            VectorMatch(
                chunk_id=record.chunk_id,
                text=record.text,
                metadata=record.metadata,
                score=self._cosine_similarity(query_embedding, record.embedding),
            )
            for record in self._load()
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def count(self) -> int:
        return len(self._load())

    def _load(self) -> list[VectorRecord]:
        if not self._path.exists():
            return []
        raw_records = json.loads(self._path.read_text(encoding="utf-8"))
        return [
            VectorRecord(
                chunk_id=str(item["chunk_id"]),
                text=str(item["text"]),
                metadata={str(key): str(value) for key, value in item["metadata"].items()},
                embedding=[float(value) for value in item["embedding"]],
            )
            for item in raw_records
        ]

    def _cosine_similarity(
        self,
        left: list[float],
        right: list[float],
    ) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0

        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
