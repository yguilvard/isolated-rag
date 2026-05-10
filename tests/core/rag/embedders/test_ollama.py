import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.rag.embedders.ollama import OllamaEmbedder
from src.core.rag.models import Chunk


def _chunk(index: int = 0) -> Chunk:
    return Chunk(document_path=Path("f.txt"), index=index, content=f"sentence {index}")


@pytest.mark.asyncio
async def test_returns_embedding_per_chunk():
    with patch("src.core.rag.embedders.ollama.OllamaEmbeddings") as MockOllamaEmbeddings:
        mock_instance = MockOllamaEmbeddings.return_value
        mock_instance.aembed_documents = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")
        embeddings = await embedder.embed([_chunk(0), _chunk(1)])

    assert len(embeddings) == 2
    assert embeddings[0].vector == [0.1, 0.2, 0.3]
    assert embeddings[0].model == "nomic-embed-text"
    assert embeddings[0].chunk.index == 0
    assert embeddings[1].vector == [0.4, 0.5, 0.6]
    assert embeddings[1].chunk.index == 1


@pytest.mark.asyncio
async def test_passes_texts_to_embed_documents():
    with patch("src.core.rag.embedders.ollama.OllamaEmbeddings") as MockOllamaEmbeddings:
        mock_instance = MockOllamaEmbeddings.return_value
        mock_instance.aembed_documents = AsyncMock(return_value=[[0.1]])

        embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")
        await embedder.embed([_chunk(0)])

        mock_instance.aembed_documents.assert_called_once_with(["sentence 0"])


@pytest.mark.asyncio
async def test_instantiates_with_correct_model_and_url():
    with patch("src.core.rag.embedders.ollama.OllamaEmbeddings") as MockOllamaEmbeddings:
        mock_instance = MockOllamaEmbeddings.return_value
        mock_instance.aembed_documents = AsyncMock(return_value=[[0.1]])

        OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")

        MockOllamaEmbeddings.assert_called_once_with(
            model="nomic-embed-text", base_url="http://localhost:11434"
        )
