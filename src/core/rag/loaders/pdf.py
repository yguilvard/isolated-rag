from pathlib import Path

import structlog
from pypdf import PdfReader

from src.core.rag.models import Document

logger = structlog.get_logger()


class PDFLoader:
    """Loads PDF files into Documents via text extraction."""

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

        # Extract text from each page and join
        reader = PdfReader(str(path))
        content = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

        logger.info("document_loaded", path=str(path), pages=len(reader.pages))
        return Document(
            path=path,
            content=content,
            mime_type="application/pdf",
        )
