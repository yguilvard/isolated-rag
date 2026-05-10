from pathlib import Path
from unittest.mock import patch

from src.core.rag.chunkers.sentence import SentenceChunker, _split_by_words
from src.core.rag.models import Document


def _doc(content: str) -> Document:
    return Document(path=Path("f.txt"), content=content, mime_type="text/plain")


def test_groups_sentences_into_chunks():
    # chunk_sentences=3, overlap=1 → step=2
    # [S1,S2,S3], [S3,S4,S5], [S5,S6] → 3 chunks
    sentences = ["S1.", "S2.", "S3.", "S4.", "S5.", "S6."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3, overlap_sentences=1)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 3
    assert chunks[0].content == "S1. S2. S3."
    assert chunks[0].index == 0
    assert chunks[1].content == "S3. S4. S5."
    assert chunks[1].index == 1
    assert chunks[2].content == "S5. S6."
    assert chunks[2].index == 2


def test_no_overlap_groups_cleanly():
    # overlap=0 preserves the original non-overlapping behaviour
    sentences = ["S1.", "S2.", "S3.", "S4.", "S5.", "S6."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3, overlap_sentences=0)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 2
    assert chunks[0].content == "S1. S2. S3."
    assert chunks[1].content == "S4. S5. S6."


def test_partial_last_chunk():
    # 4 sentences, chunk=3, overlap=1, step=2 → [S1,S2,S3], [S3,S4]
    sentences = ["S1.", "S2.", "S3.", "S4."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3, overlap_sentences=1)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 2
    assert chunks[1].content == "S3. S4."


def test_chunk_carries_document_path():
    sentences = ["Hello world."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=5)
        chunks = chunker.chunk(_doc("Hello world."))

    assert chunks[0].document_path == Path("f.txt")


# ---------------------------------------------------------------------------
# Fallback: oversized sentence → word-boundary splitting
# ---------------------------------------------------------------------------

def test_split_by_words_basic():
    # 3 words of ~5 chars each; max_chars=12 → two pieces
    text = "hello world foo bar baz"
    pieces = _split_by_words(text, max_chars=12)
    assert all(len(p) <= 12 for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_split_by_words_single_word_longer_than_limit():
    # A single very long token cannot be split further; returned as-is
    pieces = _split_by_words("superlongwordthatexceedslimit", max_chars=5)
    assert pieces == ["superlongwordthatexceedslimit"]


def test_oversized_sentence_triggers_word_split():
    # NLTK returns 1 giant "sentence" (OCR text, no punctuation).
    # With max_sentence_chars=20 it should be split into multiple pieces,
    # and those pieces then go through the sliding-window chunker.
    long_blob = "word " * 40  # 200 chars, well over max_sentence_chars=20
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=[long_blob.strip()]):
        chunker = SentenceChunker(chunk_sentences=3, overlap_sentences=0, max_sentence_chars=20)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) > 1, "Expected multiple chunks from fallback word-split"
    # Reconstruct all words and verify nothing was lost
    all_words = " ".join(c.content for c in chunks).split()
    assert all_words == long_blob.strip().split()


def test_normal_sentences_not_expanded():
    # Short sentences (< max_sentence_chars) are never split further.
    sentences = ["Short.", "Also short.", "Fine."]
    with patch("src.core.rag.chunkers.sentence.nltk.sent_tokenize", return_value=sentences):
        chunker = SentenceChunker(chunk_sentences=3, overlap_sentences=0, max_sentence_chars=500)
        chunks = chunker.chunk(_doc("irrelevant"))

    assert len(chunks) == 1
    assert chunks[0].content == "Short. Also short. Fine."
