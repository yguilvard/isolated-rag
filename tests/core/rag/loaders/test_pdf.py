import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.rag.loaders.pdf import PDFLoader


def test_loads_pdf_content(tmp_path: Path):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page one text."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("src.core.rag.loaders.pdf.PdfReader", return_value=mock_reader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    assert doc.content == "Page one text."
    assert doc.mime_type == "application/pdf"
    assert doc.path == f


def test_joins_multiple_pages(tmp_path: Path):
    page1 = MagicMock()
    page1.extract_text.return_value = "First."
    page2 = MagicMock()
    page2.extract_text.return_value = "Second."
    mock_reader = MagicMock()
    mock_reader.pages = [page1, page2]

    with patch("src.core.rag.loaders.pdf.PdfReader", return_value=mock_reader):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF fake")
        loader = PDFLoader()
        doc = loader.load(f)

    assert doc.content == "First.\nSecond."


def test_raises_on_unsupported_extension(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("text")
    loader = PDFLoader()
    with pytest.raises(ValueError, match="Unsupported extension"):
        loader.load(f)
