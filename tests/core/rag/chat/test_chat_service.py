import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.core.rag.chat.service import ChatService, RetrievedChunk


@pytest.fixture
def service():
    with patch("src.core.rag.chat.service.OllamaEmbeddings"):
        svc = ChatService(embedding_model="bge-m3", embedding_url="http://localhost:11434")
    return svc


@pytest.fixture
def conn():
    mock = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    mock.transaction = MagicMock(return_value=txn)
    mock.fetch = AsyncMock(return_value=[
        {"chunk_index": 5, "document_path": "doc.pdf", "content": "Relevant content.", "score": 0.032},
    ])
    return mock


def _make_llm(response: str = "Hypothetical passage about the topic.") -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = response
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


async def test_retrieve_without_llm_skips_hyde(service, conn):
    service._embedder.aembed_query = AsyncMock(return_value=[0.1, 0.2])
    await service.retrieve("lightgbm", conn, uuid4(), "all", 5)
    # Embedder called with the raw query
    service._embedder.aembed_query.assert_awaited_once_with("lightgbm")


async def test_retrieve_with_llm_uses_hyde_passage(service, conn):
    llm = _make_llm("LightGBM is a gradient boosting framework used for prediction tasks.")
    service._embedder.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    # Multi-word query (≥3 words) triggers HyDE
    await service.retrieve("how does lightgbm work", conn, uuid4(), "all", 5, llm=llm)

    call_arg = service._embedder.aembed_query.call_args[0][0]
    assert call_arg != "how does lightgbm work"
    assert "LightGBM" in call_arg or "gradient" in call_arg


async def test_retrieve_hyde_skipped_for_short_query(service, conn):
    llm = _make_llm("Some expansion.")
    service._embedder.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    # Single word — HyDE guard skips expansion
    await service.retrieve("lightgbm", conn, uuid4(), "all", 5, llm=llm)

    service._embedder.aembed_query.assert_awaited_once_with("lightgbm")
    llm.ainvoke.assert_not_awaited()


async def test_retrieve_hyde_falls_back_on_llm_error(service, conn):
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    service._embedder.aembed_query = AsyncMock(return_value=[0.1, 0.2])

    await service.retrieve("lightgbm", conn, uuid4(), "all", 5, llm=llm)

    # Falls back to raw query
    service._embedder.aembed_query.assert_awaited_once_with("lightgbm")


async def test_retrieve_passes_original_query_for_fts(service, conn):
    """FTS always uses the original query, not the HyDE passage."""
    llm = _make_llm("Some expanded hypothetical passage.")
    service._embedder.aembed_query = AsyncMock(return_value=[0.1, 0.2])
    uid = uuid4()

    # Multi-word to trigger HyDE, ensuring FTS still uses the original
    await service.retrieve("how does lightgbm work", conn, uid, "all", 5, llm=llm)

    fetch_args = conn.fetch.call_args[0]
    # conn.fetch(sql, $1_vector, $2_fts_text, $3_scope, $4_limit, $5_min_score)
    assert fetch_args[2] == "how does lightgbm work"


async def test_retrieve_returns_mapped_chunks(service, conn):
    service._embedder.aembed_query = AsyncMock(return_value=[0.1])
    result = await service.retrieve("q", conn, uuid4(), "all", 5)

    assert len(result) == 1
    assert isinstance(result[0], RetrievedChunk)
    assert result[0].chunk_index == 5
    assert result[0].content == "Relevant content."
    assert result[0].score == pytest.approx(0.032)


async def test_hyde_expansion_empty_response_falls_back(service, conn):
    llm = _make_llm("")  # empty response
    service._embedder.aembed_query = AsyncMock(return_value=[0.1])

    await service.retrieve("lightgbm", conn, uuid4(), "all", 5, llm=llm)

    service._embedder.aembed_query.assert_awaited_once_with("lightgbm")
