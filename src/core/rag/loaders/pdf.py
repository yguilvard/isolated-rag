import re
from pathlib import Path

import structlog
from langchain_community.document_loaders import PyPDFLoader

from src.core.rag.models import Document

logger = structlog.get_logger()


class PDFLoader:
    """Loads PDF files into Documents via LangChain's PyPDFLoader."""

    def load(self, path: Path) -> Document:
        """Extract text from all pages of a PDF file.

        Args:
            path: Path to the PDF file.

        Returns:
            A Document with extracted text content.

        Raises:
            ValueError: If the file extension is not .pdf.
        """
        # Validate supported extension
        if path.suffix != ".pdf":
            raise ValueError(f"Unsupported extension: {path.suffix!r}")

        # Load pages via LangChain (uses pypdf under the hood)
        lc_loader = PyPDFLoader(str(path))
        pages = lc_loader.load()

        # Join pages with blank line (paragraph break) so NLTK can detect
        # sentence boundaries across pages
        raw = "\n\n".join(page.page_content for page in pages)

        # Normalize PDF text: single newlines are mid-sentence line wraps,
        # not real breaks — collapse them so NLTK can find sentence boundaries
        content = self._normalize(raw)

        logger.info("document_loaded", path=str(path), pages=len(pages))
        return Document(
            path=path,
            content=content,
            mime_type="application/pdf",
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Collapse mid-line wraps while preserving paragraph boundaries."""
        # Preserve double newlines (paragraph breaks) as sentinel
        text = re.sub(r"\n{2,}", "\x00", text)
        # Collapse remaining single newlines into spaces
        text = re.sub(r"\n", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Restore paragraph breaks
        text = re.sub(r"\x00", "\n\n", text)
        return text.strip()
