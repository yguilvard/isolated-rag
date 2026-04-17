import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from src.core.rag.models import Chunk, Embedding
from src.core.rag.store.pgvector import PgVectorStore


def _embedding(index: int = 0, owner_id=None, visibility="private") -> Embedding:
    chunk = Chunk(document_path=Path("f.txt"), index=index, content=f"text {index}")
    return Embedding(
        chunk=chunk,
        vector=[0.1, 0.2, 0.3],
        model="nomic-embed-text",
        owner_id=owner_id,
        visibility=visibility,
    )


@pytest.mark.asyncio
async def test_setup_creates_extension_table_and_alters_columns():
    mock_conn = AsyncMock()
    with patch("src.core.rag.store.pgvector.asyncpg.connect", return_value=mock_conn):
        store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
        await store.setup()

    calls = [call.args[0] for call in mock_conn.execute.call_args_list]
    assert any("CREATE EXTENSION" in c for c in calls)
    assert any("CREATE TABLE" in c for c in calls)
    assert any("ALTER TABLE" in c for c in calls)
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_save_upserts_embeddings():
    mock_conn = AsyncMock()
    with patch("src.core.rag.store.pgvector.asyncpg.connect", return_value=mock_conn):
        store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
        await store.save([_embedding(0), _embedding(1)])

    assert mock_conn.execute.call_count == 2
    mock_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_save_formats_vector_as_string():
    mock_conn = AsyncMock()
    with patch("src.core.rag.store.pgvector.asyncpg.connect", return_value=mock_conn):
        store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
        await store.save([_embedding(0)])

    call_args = mock_conn.execute.call_args_list[0].args
    # vector is the 6th positional arg (index 5): SQL, path, idx, content, model, vector, owner_id, visibility
    assert call_args[5] == "[0.1,0.2,0.3]"


@pytest.mark.asyncio
async def test_save_passes_owner_id_and_visibility():
    mock_conn = AsyncMock()
    uid = uuid4()
    with patch("src.core.rag.store.pgvector.asyncpg.connect", return_value=mock_conn):
        store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
        await store.save([_embedding(0, owner_id=uid, visibility="public")])

    call_args = mock_conn.execute.call_args_list[0].args
    assert call_args[6] == uid        # owner_id
    assert call_args[7] == "public"   # visibility


@pytest.mark.asyncio
async def test_save_uses_provided_connection_without_closing():
    """When a connection is passed, save() uses it and does not close it."""
    external_conn = AsyncMock()
    store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
    await store.save([_embedding(0)], conn=external_conn)

    assert external_conn.execute.call_count == 1
    external_conn.close.assert_not_called()
