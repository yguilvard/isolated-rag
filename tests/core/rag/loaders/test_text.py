import pytest
from pathlib import Path

from src.core.rag.loaders.text import TextLoader


def test_loads_txt_file(tmp_path: Path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world.", encoding="utf-8")
    loader = TextLoader()
    doc = loader.load(f)
    assert doc.content == "Hello world."
    assert doc.mime_type == "text/plain"
    assert doc.path == f


def test_loads_md_file(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\nContent.", encoding="utf-8")
    loader = TextLoader()
    doc = loader.load(f)
    assert doc.content == "# Title\nContent."
    assert doc.mime_type == "text/markdown"


def test_raises_on_unsupported_extension(tmp_path: Path):
    f = tmp_path / "doc.xyz"
    f.write_text("data", encoding="utf-8")
    loader = TextLoader()
    with pytest.raises(ValueError, match="Unsupported extension"):
        loader.load(f)
