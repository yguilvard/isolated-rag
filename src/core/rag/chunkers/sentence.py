import nltk
import structlog

from src.core.rag.models import Chunk, Document

logger = structlog.get_logger()

# Download punkt tokenizer data on first use (silent)
nltk.download("punkt_tab", quiet=True)


class SentenceChunker:
    """Splits documents into chunks of N consecutive sentences."""

    def __init__(self, chunk_sentences: int = 5) -> None:
        """Initialize a SentenceChunker.

        Args:
            chunk_sentences: Number of sentences to include in each chunk.
        """
        self._chunk_sentences = chunk_sentences

    def chunk(self, document: Document) -> list[Chunk]:
        """Tokenize document into sentences, group into fixed-size chunks."""
        # Tokenize content into individual sentences
        sentences = nltk.sent_tokenize(document.content)

        # Group sentences into chunks of self._chunk_sentences
        chunks = []
        for i in range(0, len(sentences), self._chunk_sentences):
            group = sentences[i : i + self._chunk_sentences]
            chunks.append(
                Chunk(
                    document_path=document.path,
                    index=i // self._chunk_sentences,
                    content=" ".join(group),
                )
            )

        logger.info("document_chunked", path=str(document.path), chunks=len(chunks))
        return chunks
