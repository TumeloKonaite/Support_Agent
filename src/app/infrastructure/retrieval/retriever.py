from dataclasses import dataclass, field
from typing import Protocol

from src.app.infrastructure.retrieval.embedding import Embedder
from src.app.infrastructure.retrieval.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    """A chunk of indexed business knowledge."""

    chunk_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    """A retrieval result enriched with similarity score."""

    chunk: KnowledgeChunk
    score: float


class Retriever(Protocol):
    """Abstraction for top-k knowledge retrieval."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        """Return ranked knowledge chunks for the query."""


class VectorStoreRetriever:
    """Retriever that embeds the query and searches a vector store."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = 3,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedContext]:
        matches = self._vector_store.search(
            query_embedding=self._embedder.embed_query(query),
            top_k=top_k or self._default_top_k,
        )
        return [
            RetrievedContext(
                chunk=KnowledgeChunk(
                    chunk_id=match.chunk_id,
                    text=match.text,
                    metadata=match.metadata,
                ),
                score=match.score,
            )
            for match in matches
        ]
