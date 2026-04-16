from pathlib import Path
from unittest.mock import patch

from src.core.rag.chunkers.sentence import SentenceChunker
from src.core.rag.models import Document


def _doc(content: str) -> Document:
    return Document(path=Path("f.txt"), content=content, mime_type="text/plain")


def test_groups_sentences_into_chunks():
    sentences = ["S1.", "S2.", "S3.", "S4.", "S5.", "S6."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 2
    assert chunks[0].content == "S1. S2. S3."
    assert chunks[0].index == 0
    assert chunks[1].content == "S4. S5. S6."
    assert chunks[1].index == 1


def test_partial_last_chunk():
    sentences = ["S1.", "S2.", "S3.", "S4."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 2
    assert chunks[1].content == "S4."


def test_chunk_carries_document_path():
    sentences = ["Hello world."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=5)
        chunks = chunker.chunk(_doc("Hello world."))

    assert chunks[0].document_path == Path("f.txt")
