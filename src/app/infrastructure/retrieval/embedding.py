import math
import re
from collections.abc import Sequence
from hashlib import blake2b
from typing import Protocol

from openai import OpenAI


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Abstraction for providers that convert text into vector embeddings."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for many texts."""

    def embed_query(self, text: str) -> list[float]:
        """Return an embedding for one query text."""


class HashingEmbedder:
    """Deterministic local embedder suitable for tests and offline indexing."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            return vector

        for token in tokens:
            bucket = self._bucket_for(token)
            vector[bucket] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0.0:
            return vector

        return [value / magnitude for value in vector]

    def _bucket_for(self, token: str) -> int:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self._dimensions


class OpenAIEmbedder:
    """Embed text using the OpenAI embeddings API."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        """Return the configured embedding model."""
        return self._model

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
