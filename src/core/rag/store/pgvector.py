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

# Adds owner/visibility columns without FK (FK added later by API migration).
# Both ADD COLUMN clauses use IF NOT EXISTS, making this statement idempotent:
# when a column already exists the clause becomes a no-op and the inline CHECK
# constraint is not re-evaluated. Safe to call on every startup.
_ALTER_EMBEDDINGS_OWNER = """
    ALTER TABLE embeddings
        ADD COLUMN IF NOT EXISTS owner_id   UUID,
        ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private'
            CHECK (visibility IN ('private', 'public'))
"""

_UPSERT = """
    INSERT INTO embeddings
        (document_path, chunk_index, content, model, vector, owner_id, visibility)
    VALUES ($1, $2, $3, $4, $5::vector, $6, $7)
    ON CONFLICT (document_path, chunk_index) DO UPDATE
        SET content    = EXCLUDED.content,
            model      = EXCLUDED.model,
            vector     = EXCLUDED.vector,
            owner_id   = EXCLUDED.owner_id,
            visibility = EXCLUDED.visibility,
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
        """Create the vector extension, embeddings table, and owner columns if absent."""
        conn = await asyncpg.connect(self._dsn)
        try:
            await conn.execute(_CREATE_EXTENSION)
            await conn.execute(_CREATE_TABLE)
            await conn.execute(_ALTER_EMBEDDINGS_OWNER)
        finally:
            await conn.close()
        logger.info("store_ready")

    async def save(
        self,
        embeddings: list[Embedding],
        conn: asyncpg.Connection | None = None,
    ) -> None:
        """Upsert embeddings into the store.

        Args:
            embeddings: List of Embedding objects to persist.
            conn: Optional existing connection. If provided, caller owns it and
                  it is not closed. If None, a new connection is created and closed.
        """
        own_conn = conn is None
        if own_conn:
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
                    emb.owner_id,
                    emb.visibility,
                )
        finally:
            if own_conn:
                await conn.close()
        logger.info("embeddings_saved", count=len(embeddings))
