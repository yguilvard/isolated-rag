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

    # First call: set_config; second call: DELETE stale chunks
    calls = conn.execute.call_args_list
    assert any("set_config" in str(c) for c in calls)
    assert any("DELETE" in str(c) for c in calls)
    conn.transaction.assert_called_once()


async def test_run_deletes_existing_chunks_before_save(service, conn, tmp_path):
    doc = tmp_path / "report.txt"
    doc.write_text("Hello world.")
    uid = uuid4()

    with patch("src.core.services.ingest.TextLoader"):
        await service.run(doc, uid, "private", conn)

    delete_calls = [c for c in conn.execute.call_args_list if "DELETE" in str(c)]
    assert len(delete_calls) == 1
    # stored_doc_path falls back to str(path) when document_name is not provided
    assert str(doc) in str(delete_calls[0])


async def test_run_attaches_owner_id_and_visibility_to_embeddings(service, store, conn, tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text("Hello world.")
    uid = uuid4()

    with patch("src.core.services.ingest.TextLoader"):
        await service.run(doc, uid, "public", conn)

    saved = store.save.call_args[0][0]
    assert all(e.owner_id == uid for e in saved)
    assert all(e.visibility == "public" for e in saved)


async def test_run_applies_noise_filter_when_set(chunker, embedder, store, conn, tmp_path):
    from unittest.mock import AsyncMock as AM
    from src.core.rag.filters.heuristic import HeuristicNoiseFilter
    from src.core.rag.models import Chunk
    from pathlib import Path

    noise_filter = HeuristicNoiseFilter()
    svc = IngestService(chunker=chunker, embedder=embedder, store=store, noise_filter=noise_filter)

    # Chunker returns one good chunk and one noise chunk (9 chars — below min_length)
    good = Chunk(document_path=Path("doc.txt"), index=0, content="A" * 60)
    noise = Chunk(document_path=Path("doc.txt"), index=1, content="X" * 9)
    chunker.chunk.return_value = [good, noise]

    doc = tmp_path / "doc.txt"
    doc.write_text("...")
    with patch("src.core.services.ingest.TextLoader"):
        count = await svc.run(doc, uuid4(), "private", conn)

    # Only the good chunk reaches the embedder
    embedded_chunks = embedder.embed.call_args[0][0]
    assert len(embedded_chunks) == 1
    assert embedded_chunks[0].content == good.content
    assert count == 1


async def test_run_raises_on_unsupported_extension(service, conn, tmp_path):
    doc = tmp_path / "doc.xyz"
    doc.write_text("data")
    with pytest.raises(ValueError, match="Unsupported file type"):
        await service.run(doc, uuid4(), "private", conn)
