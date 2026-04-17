import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.rag.loaders.pdf import PDFLoader


def _lc_page(text: str) -> MagicMock:
    """Return a mock LangChain Document with .page_content set."""
    page = MagicMock()
    page.page_content = text
    return page


def test_loads_pdf_content(tmp_path: Path):
    mock_loader = MagicMock()
    mock_loader.load.return_value = [_lc_page("Page one text.")]

    with patch("src.core.rag.loaders.pdf.PyPDFLoader", return_value=mock_loader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    assert doc.content == "Page one text."
    assert doc.mime_type == "application/pdf"
    assert doc.path == f


def test_joins_multiple_pages(tmp_path: Path):
    mock_loader = MagicMock()
    mock_loader.load.return_value = [_lc_page("First."), _lc_page("Second.")]

    with patch("src.core.rag.loaders.pdf.PyPDFLoader", return_value=mock_loader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    # Pages are separated by a blank line (paragraph break) so NLTK
    # can detect sentence boundaries across pages
    assert doc.content == "First.\n\nSecond."


def test_raises_on_unsupported_extension(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("text")
    loader = PDFLoader()
    with pytest.raises(ValueError, match="Unsupported extension"):
        loader.load(f)
