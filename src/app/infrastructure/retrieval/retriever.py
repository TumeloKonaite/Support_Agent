from dataclasses import dataclass, field
from typing import Protocol

from src.app.infrastructure.retrieval.embedding import TOKEN_PATTERN, Embedder
from src.app.infrastructure.retrieval.vector_store import VectorStore


LEXICAL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "can",
        "do",
        "does",
        "for",
        "help",
        "how",
        "i",
        "is",
        "me",
        "my",
        "of",
        "the",
        "to",
        "what",
        "when",
        "where",
        "with",
        "you",
        "your",
    }
)
LEXICAL_RERANK_CANDIDATE_MULTIPLIER = 5


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
    original_score: float | None = None
    reranker_score: float | None = None


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
        requested_top_k = top_k or self._default_top_k
        matches = self._vector_store.search(
            query_embedding=self._embedder.embed_query(query),
            top_k=requested_top_k * LEXICAL_RERANK_CANDIDATE_MULTIPLIER,
        )
        results = [
            RetrievedContext(
                chunk=KnowledgeChunk(
                    chunk_id=match.chunk_id,
                    text=match.text,
                    metadata=match.metadata,
                ),
                score=max(match.score, self._lexical_overlap_score(query, match.text)),
            )
            for match in matches
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:requested_top_k]

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        query_tokens = self._content_tokens(query)
        if not query_tokens:
            return 0.0

        text_tokens = self._content_tokens(text)
        if not text_tokens:
            return 0.0

        return len(query_tokens & text_tokens) / len(query_tokens)

    def _content_tokens(self, text: str) -> set[str]:
        return {
            self._normalize_token(token)
            for token in TOKEN_PATTERN.findall(text.lower())
            if token not in LEXICAL_STOPWORDS
        }

    def _normalize_token(self, token: str) -> str:
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token
