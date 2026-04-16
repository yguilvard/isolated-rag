import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.core.rag.models import Chunk, Embedding
from src.core.rag.store.pgvector import PgVectorStore


def _embedding(index: int = 0) -> Embedding:
    chunk = Chunk(document_path=Path("f.txt"), index=index, content=f"text {index}")
    return Embedding(chunk=chunk, vector=[0.1, 0.2, 0.3], model="nomic-embed-text")


@pytest.mark.asyncio
async def test_setup_creates_extension_and_table():
    mock_conn = AsyncMock()
    with patch("src.core.rag.store.pgvector.asyncpg.connect", return_value=mock_conn):
        store = PgVectorStore(dsn="postgresql://rag:pw@localhost/rag_db")
        await store.setup()

    calls = [call.args[0] for call in mock_conn.execute.call_args_list]
    assert any("CREATE EXTENSION" in c for c in calls)
    assert any("CREATE TABLE" in c for c in calls)
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
    # Last argument should be the vector formatted as "[0.1,0.2,0.3]"
    assert call_args[-1] == "[0.1,0.2,0.3]"
