import json
import re
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.rag.models import Chunk

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Inspect only the first N chunks — TOC is always at the document start.
_DEFAULT_MAX_CHUNKS = 30


def _parse_indices(raw: str, valid: set[int]) -> list[int]:
    """Parse and validate an LLM response as a list of chunk indices.

    Strips optional markdown code fences, then enforces:
    - Valid JSON
    - Top-level value is a list
    - Every element is an integer present in *valid*

    Args:
        raw: Raw text response from the LLM.
        valid: Set of chunk indices that were sent for inspection.

    Returns:
        Validated list of TOC chunk indices.

    Raises:
        ValueError: On any structural or range violation.
    """
    # Strip markdown code fences that some models add despite instructions
    text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    parsed = json.loads(text)

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")

    result: list[int] = []
    for item in parsed:
        if not isinstance(item, int):
            raise ValueError(f"Non-integer element: {item!r}")
        if item not in valid:
            raise ValueError(f"Index {item} outside inspected window {sorted(valid)}")
        result.append(item)

    return result


class LLMTOCFilter:
    """Filters table-of-contents chunks using an LLM chat completion.

    Sends the first *max_chunks* chunks to a generative model via the
    chat API (system + user roles).  The LLM returns a JSON array of
    indices it identifies as TOC entries; those chunks are dropped.

    Two prompt-injection defences are applied:

    1. **Chat role separation** — task instructions live in the system
       message; user-supplied context is confined to the user message
       inside explicit ``<user_context>`` delimiters.
    2. **Structural output validation** — the response is parsed as
       strict JSON.  Every element must be an integer present in the
       inspected window.  Any violation triggers a warning and the safe
       default: keep all chunks.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        max_chunks: int = _DEFAULT_MAX_CHUNKS,
    ) -> None:
        """Initialise the filter.

        Args:
            model: Ollama generative model name (e.g. ``llama3.2``).
            base_url: Base URL of the Ollama server.
            max_chunks: Maximum number of chunks to inspect. Chunks
                beyond this limit are passed through unchanged.
        """
        self._max_chunks = max_chunks
        # temperature=0 for deterministic classification output
        self._llm = ChatOllama(model=model, base_url=base_url, temperature=0)

        # Load Jinja2 templates from the prompts/ directory
        env = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)
        self._system_tmpl = env.get_template("toc_system.j2")
        self._user_tmpl = env.get_template("toc_user.j2")

    async def filter(self, chunks: list[Chunk], *, context: str = "") -> list[Chunk]:
        """Remove TOC chunks identified by the LLM.

        Only the first *max_chunks* chunks are sent for inspection; the
        rest are passed through without classification.  On any LLM or
        parsing error, all chunks are returned unchanged.

        Args:
            chunks: Ordered list of document chunks.
            context: Optional user-supplied description of the document,
                embedded in the user message inside ``<user_context>``
                delimiters and explicitly instructed not to be followed
                as instructions.

        Returns:
            Chunks with TOC entries removed, or the original list on error.
        """
        if not chunks:
            return chunks

        # Inspect only the leading window where TOC sections appear
        window = chunks[: self._max_chunks]
        valid_indices = {c.index for c in window}

        # Render prompt templates
        system_content = self._system_tmpl.render()
        user_content = self._user_tmpl.render(chunks=window, user_context=context.strip())

        try:
            # Defense 2: system role carries instructions; user role carries data
            response = await self._llm.ainvoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(content=user_content),
                ]
            )
            raw: str = response.content  # type: ignore[assignment]

            # Defense 1: strict structural validation — any violation raises
            toc_indices = _parse_indices(raw, valid_indices)

        except Exception as exc:
            # Safe default: keep all chunks rather than silently dropping content
            logger.warning(
                "toc_filter_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return chunks

        if not toc_indices:
            logger.info("toc_filter_no_toc_found", inspected=len(window))
            return chunks

        toc_set = set(toc_indices)
        kept = [c for c in chunks if c.index not in toc_set]

        logger.info(
            "toc_filter_removed",
            removed=len(chunks) - len(kept),
            toc_indices=toc_indices,
            total_chunks=len(chunks),
        )
        return kept
