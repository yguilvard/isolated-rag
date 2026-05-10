from pathlib import Path

import structlog

from src.core.rag.models import Document

logger = structlog.get_logger()

_MIME_TYPES: dict[str, str] = {
    ".txt": "text/plain",
    ".md": "text/markdown",
}


class TextLoader:
    """Loads plain text and Markdown files into Documents."""

    def load(self, path: Path) -> Document:
        """Load a .txt or .md file from disk."""
        # Validate supported extension
        if path.suffix not in _MIME_TYPES:
            raise ValueError(f"Unsupported extension: {path.suffix!r}")

        # Read file content
        content = path.read_text(encoding="utf-8")

        logger.info("document_loaded", path=str(path), chars=len(content))
        return Document(
            path=path,
            content=content,
            mime_type=_MIME_TYPES[path.suffix],
        )
