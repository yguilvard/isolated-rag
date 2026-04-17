import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.core.rag.models import Chunk, Embedding
from src.core.services.ingest import IngestService


def _make_embedding(path: Path = Path("doc.txt"), index: int = 0) -> Embedding:
    chunk = Chunk(document_path=path, index=index, content="Hello world.")
    return Embedding(chunk=chunk, vector=[0.1, 0.2], model="nomic-embed-text")


@pytest.fixture
def chunker():
    mock = MagicMock()
    mock.chunk.return_value = [MagicMock()]
    return mock


@pytest.fixture
def embedder():
    mock = MagicMock()
    mock.embed = AsyncMock(return_value=[_make_embedding()])
    return mock


@pytest.fixture
def store():
    mock = MagicMock()
    mock.save = AsyncMock()
    return mock


@pytest.fixture
def service(chunker, embedder, store):
    return IngestService(chunker=chunker, embedder=embedder, store=store)


@pytest.fixture
def conn():
    mock = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    # Use MagicMock (not AsyncMock) because asyncpg.Connection.transaction()
    # is synchronous — it returns a Transaction context manager, not a coroutine.
    mock.transaction = MagicMock(return_value=txn)
    return mock


async def test_run_returns_chunk_count(service, conn, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Hello world.")
    uid = uuid4()

    with patch("src.core.services.ingest.TextLoader") as MockLoader:
        mock_doc = MagicMock()
        MockLoader.return_value.load.return_value = mock_doc
        count = await service.run(doc, uid, "private", conn)

    assert count == 1


async def test_run_sets_rls_config_inside_transaction(service, conn, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Hello world.")
    uid = uuid4()

    with patch("src.core.services.ingest.TextLoader"):
        await service.run(doc, uid, "private", conn)

    conn.execute.assert_called_once_with(
        "SELECT set_config('app.current_user_id', $1, true)", str(uid)
    )
    conn.transaction.assert_called_once()


async def test_run_attaches_owner_id_and_visibility_to_embeddings(service, store, conn, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Hello world.")
    uid = uuid4()

    with patch("src.core.services.ingest.TextLoader"):
        await service.run(doc, uid, "public", conn)

    saved = store.save.call_args[0][0]
    assert all(e.owner_id == uid for e in saved)
    assert all(e.visibility == "public" for e in saved)


async def test_run_raises_on_unsupported_extension(service, conn, tmp_path):
    doc = tmp_path / "doc.xyz"
    doc.write_text("data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        await service.run(doc, uuid4(), "private", conn)
