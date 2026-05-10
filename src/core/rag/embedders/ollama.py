import structlog
from langchain_ollama import OllamaEmbeddings

from src.core.rag.models import Chunk, Embedding

logger = structlog.get_logger()


class OllamaEmbedder:
    """Embeds text chunks via LangChain's OllamaEmbeddings."""

    def __init__(self, model: str, base_url: str) -> None:
        """Initialize the embedder.

        Args:
            model: Ollama model name (e.g. nomic-embed-text).
            base_url: Base URL of the Ollama server.
        """
        self._model = model
        self._lc = OllamaEmbeddings(model=model, base_url=base_url)

    async def embed(self, chunks: list[Chunk]) -> list[Embedding]:
        """Embed all chunks in a single batched call to Ollama.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List of Embedding objects with vectors from Ollama.
        """
        # Batch-embed all chunk texts via LangChain async interface
        texts = [chunk.content for chunk in chunks]
        vectors: list[list[float]] = await self._lc.aembed_documents(texts)

        embeddings = [
            Embedding(chunk=chunk, vector=vector, model=self._model)
            for chunk, vector in zip(chunks, vectors)
        ]

        logger.info("chunks_embedded", count=len(embeddings), model=self._model)
        return embeddings
