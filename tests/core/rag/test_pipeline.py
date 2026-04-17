import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.core.rag.models import Chunk, Document, Embedding
from src.core.rag.pipeline import IngestionPipeline


def _doc() -> Document:
    return Document(path=Path("f.txt"), content="Hello.", mime_type="text/plain")


def _chunk() -> Chunk:
    return Chunk(document_path=Path("f.txt"), index=0, content="Hello.")


def _embedding() -> Embedding:
    return Embedding(chunk=_chunk(), vector=[0.1], model="nomic-embed-text")


@pytest.mark.asyncio
async def test_pipeline_calls_all_stages_in_order():
    loader = MagicMock()
    loader.load.return_value = _doc()

    chunker = MagicMock()
    chunker.chunk.return_value = [_chunk()]

    embedder = AsyncMock()
    embedder.embed.return_value = [_embedding()]

    store = AsyncMock()
    store.save.return_value = None

    pipeline = IngestionPipeline(loader, chunker, embedder, store)
    await pipeline.run(Path("f.txt"))

    loader.load.assert_called_once_with(Path("f.txt"))
    chunker.chunk.assert_called_once()
    embedder.embed.assert_called_once()
    store.save.assert_called_once()


@pytest.mark.asyncio
async def test_pipeline_passes_document_to_chunker():
    doc = _doc()
    loader = MagicMock(load=MagicMock(return_value=doc))
    chunker = MagicMock(chunk=MagicMock(return_value=[_chunk()]))
    embedder = AsyncMock(embed=AsyncMock(return_value=[_embedding()]))
    store = AsyncMock(save=AsyncMock())

    pipeline = IngestionPipeline(loader, chunker, embedder, store)
    await pipeline.run(Path("f.txt"))

    chunker.chunk.assert_called_once_with(doc)


@pytest.mark.asyncio
async def test_pipeline_passes_embeddings_to_store():
    embeddings = [_embedding()]
    loader = MagicMock(load=MagicMock(return_value=_doc()))
    chunker = MagicMock(chunk=MagicMock(return_value=[_chunk()]))
    embedder = AsyncMock(embed=AsyncMock(return_value=embeddings))
    store = AsyncMock(save=AsyncMock())

    pipeline = IngestionPipeline(loader, chunker, embedder, store)
    await pipeline.run(Path("f.txt"))

    store.save.assert_called_once_with(embeddings)
