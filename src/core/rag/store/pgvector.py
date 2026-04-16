import asyncpg
import structlog

from src.core.rag.models import Embedding

logger = structlog.get_logger()

_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS embeddings (
        id          SERIAL PRIMARY KEY,
        document_path TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        model       TEXT NOT NULL,
        vector      vector,
        created_at  TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (document_path, chunk_index)
    )
"""

_UPSERT = """
    INSERT INTO embeddings (document_path, chunk_index, content, model, vector)
    VALUES ($1, $2, $3, $4, $5::vector)
    ON CONFLICT (document_path, chunk_index) DO UPDATE
        SET content    = EXCLUDED.content,
            model      = EXCLUDED.model,
            vector     = EXCLUDED.vector,
            created_at = NOW()
"""


class PgVectorStore:
    """Persists embeddings to a pgvector-enabled PostgreSQL table."""

    def __init__(self, dsn: str) -> None:
        """Initialize the store.

        Args:
            dsn: asyncpg-compatible PostgreSQL DSN string.
        """
        self._dsn = dsn

    async def setup(self) -> None:
        """Create the vector extension and embeddings table if absent."""
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(_CREATE_EXTENSION)
            await conn.execute(_CREATE_TABLE)
        finally:
            await conn.close()
        logger.info("store_ready")

    async def save(self, embeddings: list[Embedding]) -> None:
        """Upsert embeddings into the store.

        Args:
            embeddings: List of Embedding objects to persist.
        """
        conn = await asyncpg.connect(self._dsn)
        try:
            for emb in embeddings:
                # Format vector list as pgvector string literal "[v1,v2,...]"
                vector_str = "[" + ",".join(str(v) for v in emb.vector) + "]"
                await conn.execute(
                    _UPSERT,
                    str(emb.chunk.document_path),
                    emb.chunk.index,
                    emb.chunk.content,
                    emb.model,
                    vector_str,
                )
        finally:
            await conn.close()
        logger.info("embeddings_saved", count=len(embeddings))
