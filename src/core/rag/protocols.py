from pathlib import Path
from typing import Protocol

import asyncpg

from src.core.rag.models import Chunk, Document, Embedding


class Loader(Protocol):
    """Loads a file from disk into a Document."""

    def load(self, path: Path) -> Document: ...


class Chunker(Protocol):
    """Splits a Document into sentence-grouped Chunks."""

    def chunk(self, document: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    """Converts Chunks into Embeddings via an embedding model."""

    async def embed(self, chunks: list[Chunk]) -> list[Embedding]: ...


class Store(Protocol):
    """Persists Embeddings to a vector store."""

    async def setup(self) -> None: ...

    async def save(
        self,
        embeddings: list[Embedding],
        conn: asyncpg.Connection | None = None,
    ) -> None: ...
