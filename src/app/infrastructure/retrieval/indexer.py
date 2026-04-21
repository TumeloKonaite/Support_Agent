import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.app.core.config import Settings, get_settings
from src.app.infrastructure.retrieval.embedding import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
)
from src.app.infrastructure.retrieval.retriever import (
    KnowledgeChunk,
    RetrievedContext,
    VectorStoreRetriever,
)
from src.app.infrastructure.retrieval.vector_store import JsonVectorStore, VectorRecord


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """Raw knowledge document loaded from the data directory."""

    source: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Chunking parameters used during indexing."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")


class KnowledgeIndexer:
    """Load, chunk, embed, and persist business knowledge."""

    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        vector_store: JsonVectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._data_dir = data_dir
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunking = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_documents(self) -> list[KnowledgeDocument]:
        """Load knowledge content from the configured data directory."""
        return load_documents(self._data_dir)

    def chunk_documents(
        self,
        documents: list[KnowledgeDocument],
    ) -> list[KnowledgeChunk]:
        """Split loaded documents into overlapping chunks."""
        return chunk_documents(documents, self._chunking)

    def index(self) -> list[KnowledgeChunk]:
        """Build and persist a fresh vector index from the data directory."""
        documents = self.load_documents()
        chunks = self.chunk_documents(documents)
        embeddings = self._embedder.embed_texts([chunk.text for chunk in chunks])
        records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        self._vector_store.upsert(records)
        return chunks


def build_embedder(settings: Settings) -> Embedder:
    """Build the configured embedder implementation."""
    provider = settings.retrieval_embedding_provider.lower()
    if provider == "openai":
        if settings.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY must be configured when retrieval uses OpenAI embeddings"
            )
        return OpenAIEmbedder(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.retrieval_embedding_model,
        )
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unsupported retrieval embedding provider: {provider}")


def build_indexer(settings: Settings | None = None) -> KnowledgeIndexer:
    """Construct an indexer from application settings."""
    resolved_settings = settings or get_settings()
    return KnowledgeIndexer(
        data_dir=resolved_settings.content_data_dir,
        embedder=build_embedder(resolved_settings),
        vector_store=JsonVectorStore(resolved_settings.retrieval_vector_store_path),
        chunk_size=resolved_settings.retrieval_chunk_size,
        chunk_overlap=resolved_settings.retrieval_chunk_overlap,
    )


def run_query(indexer: KnowledgeIndexer, query: str, top_k: int) -> list[RetrievedContext]:
    """Execute retrieval against the indexed knowledge base."""
    retriever = VectorStoreRetriever(
        embedder=indexer._embedder,
        vector_store=indexer._vector_store,
        default_top_k=top_k,
    )
    return retriever.retrieve(query=query, top_k=top_k)


def load_documents(data_dir: Path) -> list[KnowledgeDocument]:
    """Load supported knowledge documents from a data directory."""
    documents: list[KnowledgeDocument] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir() or not _is_supported(path):
            continue
        if "conversations" in path.parts:
            continue
        documents.extend(_load_path(path))
    return documents


def chunk_documents(
    documents: list[KnowledgeDocument],
    config: ChunkingConfig,
) -> list[KnowledgeChunk]:
    """Split documents into overlapping chunks with source metadata."""
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        for index, chunk_text in enumerate(
            _chunk_text(document.content, config),
            start=1,
        ):
            metadata = dict(document.metadata)
            metadata["source"] = document.source
            metadata["chunk_index"] = str(index)
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.source}::chunk-{index}",
                    text=chunk_text,
                    metadata=metadata,
                )
            )
    return chunks


def _load_path(path: Path) -> list[KnowledgeDocument]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _flatten_json(path, payload)

    return [
        KnowledgeDocument(
            source=path.as_posix(),
            content=path.read_text(encoding="utf-8"),
            metadata={"file_type": path.suffix.lstrip(".")},
        )
    ]


def _flatten_json(path: Path, payload: Any) -> list[KnowledgeDocument]:
    documents: list[KnowledgeDocument] = []

    def visit(value: Any, breadcrumb: list[str]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(nested, [*breadcrumb, str(key)])
            return
        if isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, [*breadcrumb, str(index)])
            return
        if value is None:
            return

        label = " > ".join(_prettify_path_part(part) for part in breadcrumb)
        documents.append(
            KnowledgeDocument(
                source=path.as_posix(),
                content=f"{label}: {value}",
                metadata={
                    "file_type": "json",
                    "path": ".".join(breadcrumb),
                },
            )
        )

    visit(payload, [])
    return documents


def _chunk_text(text: str, config: ChunkingConfig) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= config.chunk_size:
        return [stripped]

    chunks: list[str] = []
    start = 0
    step = config.chunk_size - config.chunk_overlap
    while start < len(stripped):
        end = start + config.chunk_size
        chunks.append(stripped[start:end].strip())
        if end >= len(stripped):
            break
        start += step
    return chunks


def _is_supported(path: Path) -> bool:
    return path.suffix in {".json", ".md", ".txt"}


def _prettify_path_part(value: str) -> str:
    return value.replace("_", " ")


def main() -> None:
    """Index configured data and optionally execute a test retrieval query."""
    parser = argparse.ArgumentParser(description="Build and test the local retrieval index.")
    parser.add_argument("--query", help="Optional query to test after indexing.")
    args = parser.parse_args()

    settings = get_settings()
    indexer = build_indexer(settings)
    chunks = indexer.index()
    print(f"Indexed {len(chunks)} knowledge chunks into {settings.retrieval_vector_store_path}")

    if not args.query:
        return

    for result in run_query(indexer, args.query, settings.retrieval_top_k):
        print(f"[score={result.score:.3f}] {result.chunk.chunk_id}")
        print(result.chunk.text)
        print()


if __name__ == "__main__":
    main()
