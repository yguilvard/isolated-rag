import nltk
import structlog

from src.core.rag.models import Chunk, Document

logger = structlog.get_logger()

# Download punkt tokenizer data on first use (silent)
nltk.download("punkt_tab", quiet=True)

# Sentences longer than this character limit are treated as unsplit prose
# (e.g. OCR artefacts, no punctuation) and split on word boundaries instead.
_MAX_SENTENCE_CHARS = 500


def _split_by_words(text: str, max_chars: int) -> list[str]:
    """Split *text* into pieces of at most *max_chars* characters.

    Splits only at whitespace boundaries so no word is truncated.

    Args:
        text: Input text to split.
        max_chars: Maximum character length per piece.

    Returns:
        List of non-empty text pieces.
    """
    pieces: list[str] = []
    words = text.split()
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 for the space that would precede the word
        added = len(word) + (1 if current else 0)
        if current and current_len + added > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added

    if current:
        pieces.append(" ".join(current))

    return pieces


class SentenceChunker:
    """Splits documents into overlapping chunks of N consecutive sentences.

    Each chunk overlaps with the previous one by `overlap_sentences` sentences
    so that context at chunk boundaries is not lost during retrieval.

    When NLTK sentence tokenisation yields a single oversized token (common
    with OCR'd or non-punctuated text), each token is further split on word
    boundaries using `max_sentence_chars` as the target piece size.
    """

    def __init__(
        self,
        chunk_sentences: int = 5,
        overlap_sentences: int = 1,
        max_sentence_chars: int = _MAX_SENTENCE_CHARS,
    ) -> None:
        """Initialize a SentenceChunker.

        Args:
            chunk_sentences: Number of sentences in each chunk.
            overlap_sentences: Number of sentences from the previous chunk
                to prepend to the next, for cross-boundary context.
            max_sentence_chars: Maximum character length for a single NLTK
                token before it is split further on word boundaries.
        """
        self._chunk_sentences = chunk_sentences
        self._overlap = min(overlap_sentences, chunk_sentences - 1)
        self._max_sentence_chars = max_sentence_chars

    def chunk(self, document: Document) -> list[Chunk]:
        """Tokenize document into sentences, group into overlapping chunks.

        Paragraph boundaries (blank lines) are treated as hard sentence breaks
        so that structured content without terminal punctuation — QCM exams,
        numbered lists, OCR output — is not collapsed into a single sentence by
        NLTK.  Within each paragraph, NLTK further splits on ``.?!``.  Any
        remaining token longer than *max_sentence_chars* is split on word
        boundaries.
        """
        # Split on every newline so that each extracted line — question,
        # answer option, bullet point — becomes a discrete tokenisation unit.
        # NLTK ignores newlines and collapses the entire document into one
        # "sentence" when no terminal punctuation (.?!) is present (common for
        # QCM exams, numbered lists, form fields).  Running NLTK per line still
        # lets it further split lines that contain multiple sentences.
        lines = [line.strip() for line in document.content.split("\n") if line.strip()]
        if not lines:
            return []

        raw_sentences: list[str] = []
        for line in lines:
            raw_sentences.extend(nltk.sent_tokenize(line))

        if not raw_sentences:
            return []

        # Expand oversized tokens into word-boundary pieces
        sentences: list[str] = []
        for sent in raw_sentences:
            if len(sent) > self._max_sentence_chars:
                sentences.extend(_split_by_words(sent, self._max_sentence_chars))
            else:
                sentences.append(sent)

        if not sentences:
            return []

        # Slide a window of chunk_sentences with step = chunk_sentences - overlap
        step = max(1, self._chunk_sentences - self._overlap)
        chunks = []
        idx = 0
        for start in range(0, len(sentences), step):
            group = sentences[start : start + self._chunk_sentences]
            chunks.append(
                Chunk(
                    document_path=document.path,
                    index=idx,
                    content=" ".join(group),
                )
            )
            idx += 1
            # Stop if the last sentence of this group is the last sentence overall
            if start + self._chunk_sentences >= len(sentences):
                break

        logger.info(
            "document_chunked",
            path=str(document.path),
            lines=len(lines),
            nltk_sentences=len(raw_sentences),
            units=len(sentences),
            chunks=len(chunks),
            word_split_fallback=len(sentences) != len(raw_sentences),
        )
        return chunks
