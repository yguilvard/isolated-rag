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

    # Pages are separated by a blank line so the chunker can treat them as
    # discrete units even when no terminal punctuation is present.
    assert doc.content == "First.\n\nSecond."


def test_preserves_single_newlines_within_page(tmp_path: Path):
    """Single newlines (line breaks within a page) must not be collapsed.

    Collapsing them into spaces merges every line of a QCM exam or numbered
    list into one blob, causing the chunker to produce a single chunk for the
    whole document.
    """
    mock_loader = MagicMock()
    mock_loader.load.return_value = [_lc_page("Line 1\nLine 2\nLine 3")]

    with patch("src.core.rag.loaders.pdf.PyPDFLoader", return_value=mock_loader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    assert doc.content == "Line 1\nLine 2\nLine 3"


def test_collapses_excess_blank_lines(tmp_path: Path):
    """Three or more consecutive newlines are reduced to a single blank line."""
    mock_loader = MagicMock()
    mock_loader.load.return_value = [_lc_page("A\n\n\n\nB")]

    with patch("src.core.rag.loaders.pdf.PyPDFLoader", return_value=mock_loader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    assert doc.content == "A\n\nB"


def test_raises_on_unsupported_extension(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("text")
    loader = PDFLoader()
    with pytest.raises(ValueError, match="Unsupported extension"):
        loader.load(f)
