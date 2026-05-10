import pytest
from pathlib import Path

from src.core.rag.filters.heuristic import HeuristicNoiseFilter
from src.core.rag.models import Chunk


def _chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(document_path=Path("doc.pdf"), index=index, content=content)


@pytest.fixture
def f() -> HeuristicNoiseFilter:
    return HeuristicNoiseFilter(min_length=50, max_dot_ratio=0.40)


async def test_clean_chunk_is_kept(f):
    chunk = _chunk("Le modèle LightGBM est entraîné sur les données historiques de Vélib.")
    assert await f.filter([chunk]) == [chunk]


async def test_dot_heavy_chunk_is_dropped(f):
    chunk = _chunk("LightGBM" + "." * 60 + "29")
    assert await f.filter([chunk]) == []


async def test_short_chunk_is_dropped(f):
    assert await f.filter([_chunk("Page 28 LightGBM")]) == []


async def test_empty_input_returns_empty(f):
    assert await f.filter([]) == []


async def test_mixed_list_keeps_only_clean(f):
    good = _chunk("Le modèle est évalué par station avec MAE et RMSE.", index=0)
    toc = _chunk("Entraînement des modèles" + "." * 50, index=1)
    stub = _chunk("Page 14", index=2)
    assert await f.filter([good, toc, stub]) == [good]


async def test_dot_ratio_exactly_at_threshold_is_kept(f):
    chunk = _chunk("a" * 60 + "." * 40)  # ratio = 0.40, not strictly > 0.40
    assert await f.filter([chunk]) == [chunk]


async def test_dot_ratio_just_above_threshold_is_dropped(f):
    chunk = _chunk("a" * 59 + "." * 41)  # ratio ≈ 0.41
    assert await f.filter([chunk]) == []


async def test_context_arg_is_ignored(f):
    chunk = _chunk("Le modèle LightGBM est entraîné sur les données historiques de Vélib.")
    assert await f.filter([chunk], context="ignored") == [chunk]


async def test_custom_thresholds():
    strict = HeuristicNoiseFilter(min_length=100, max_dot_ratio=0.10)
    chunk = _chunk("a" * 80)
    assert await strict.filter([chunk]) == []
    assert await HeuristicNoiseFilter().filter([chunk]) == [chunk]
