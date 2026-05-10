import structlog

from src.core.rag.models import Chunk

logger = structlog.get_logger()

_DEFAULT_MIN_LENGTH: int = 50
_DEFAULT_MAX_DOT_RATIO: float = 0.40


def _is_noise(chunk: Chunk, min_length: int, max_dot_ratio: float) -> bool:
    text = chunk.content
    if len(text) < min_length:
        return True
    return text.count(".") / len(text) > max_dot_ratio


class HeuristicNoiseFilter:
    """Drops low-quality chunks using fast, deterministic heuristics.

    Two rules applied to every chunk:

    1. **Minimum length** — chunks shorter than *min_length* chars carry no
       meaningful semantic signal (page headers, cross-page fragments, JSON
       punctuation artifacts).

    2. **Dot-leader density** — chunks whose dot-character ratio exceeds
       *max_dot_ratio* are table-of-contents or section-list entries whose
       embeddings distort cosine similarity for topic queries.
    """

    def __init__(
        self,
        min_length: int = _DEFAULT_MIN_LENGTH,
        max_dot_ratio: float = _DEFAULT_MAX_DOT_RATIO,
    ) -> None:
        self._min_length = min_length
        self._max_dot_ratio = max_dot_ratio

    async def filter(self, chunks: list[Chunk], *, context: str = "") -> list[Chunk]:
        kept = [c for c in chunks if not _is_noise(c, self._min_length, self._max_dot_ratio)]
        removed = len(chunks) - len(kept)
        if removed:
            logger.info(
                "heuristic_filter_removed",
                removed=removed,
                total=len(chunks),
                kept=len(kept),
            )
        return kept
