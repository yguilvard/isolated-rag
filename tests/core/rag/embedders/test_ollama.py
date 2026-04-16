import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.rag.embedders.ollama import OllamaEmbedder
from src.core.rag.models import Chunk


def _chunk(index: int = 0) -> Chunk:
    return Chunk(document_path=Path("f.txt"), index=index, content=f"sentence {index}")


@pytest.mark.asyncio
async def test_returns_embedding_per_chunk():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("src.core.rag.embedders.ollama.httpx.AsyncClient", return_value=mock_client):
        embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")
        embeddings = await embedder.embed([_chunk(0), _chunk(1)])

    assert len(embeddings) == 2
    assert embeddings[0].vector == [0.1, 0.2, 0.3]
    assert embeddings[0].model == "nomic-embed-text"
    assert embeddings[0].chunk.index == 0


@pytest.mark.asyncio
async def test_posts_to_correct_url():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1]}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("src.core.rag.embedders.ollama.httpx.AsyncClient", return_value=mock_client):
        embedder = OllamaEmbedder(model="nomic-embed-text", base_url="http://localhost:11434")
        await embedder.embed([_chunk()])

    mock_client.post.assert_called_once_with(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": "sentence 0"},
        timeout=60.0,
    )
