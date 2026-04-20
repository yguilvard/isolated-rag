from pathlib import Path
from typing import Literal
from uuid import UUID

import asyncpg
import structlog

from src.core.rag.loaders.pdf import PDFLoader
from src.core.rag.loaders.text import TextLoader
from src.core.rag.models import Chunk, Embedding
from src.core.rag.protocols import Chunker, Embedder, Store

logger = structlog.get_logger()

_LOADER_MAP = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


class IngestService:
    """Orchestrates document ingestion with per-user RLS context."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: Store,
    ) -> None:
        """Initialize with pipeline components.

        Args:
            chunker: Splits documents into chunks.
            embedder: Embeds chunks into vectors.
            store: Persists embeddings (any Store protocol implementer).
        """
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    async def run(
        self,
        path: Path,
        user_id: UUID,
        visibility: Literal["private", "public"],
        conn: asyncpg.Connection,
        *,
        chunker: Chunker | None = None,
        document_name: str | None = None,
    ) -> int:
        """Ingest a document for a specific user.

        Loads, chunks, embeds, and persists the document with owner_id and
        visibility set. RLS is activated inside a transaction via set_config
        so it does not leak to subsequent requests on the same pooled connection.

        Args:
            path: Path to the document file (.pdf, .txt, or .md).
            user_id: UUID of the authenticated user who owns this document.
            visibility: "private" (owner only) or "public" (all users).
            conn: asyncpg connection used for RLS-controlled persistence.
            chunker: Optional per-request chunker override (e.g. custom
                chunk_sentences from a form field).
            document_name: Human-readable name stored as the chunk's
                document_path. When omitted, the on-disk path is used.

        Returns:
            Number of chunks ingested.

        Raises:
            ValueError: If the file extension is not supported.
        """
        # Validate supported extension
        suffix = path.suffix.lower()
        if suffix not in _LOADER_MAP:
            raise ValueError(f"Unsupported file type: {suffix!r}. Supported: .pdf, .txt, .md")

        # Select loader by extension and load document
        loader = _LOADER_MAP[suffix]()
        document = loader.load(path)
        active_chunker = chunker if chunker is not None else self._chunker
        chunks = active_chunker.chunk(document)

        # Rewrite chunk document_path to the human-readable name when provided
        if document_name:
            doc_path = Path(document_name)
            chunks = [
                Chunk(document_path=doc_path, index=c.index, content=c.content)
                for c in chunks
            ]
        embeddings_raw = await self._embedder.embed(chunks)

        # Attach ownership metadata to each embedding
        embeddings = [
            Embedding(
                chunk=e.chunk,
                vector=e.vector,
                model=e.model,
                owner_id=user_id,
                visibility=visibility,
            )
            for e in embeddings_raw
        ]

        # Set RLS session variable inside a transaction (LOCAL scope) and save
        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)", str(user_id)
            )
            await self._store.save(embeddings, conn=conn)

        logger.info(
            "ingest_complete",
            path=str(path),
            chunks=len(embeddings),
            user_id=str(user_id),
            visibility=visibility,
        )
        return len(embeddings)
