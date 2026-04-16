import httpx
import structlog

from src.core.rag.models import Chunk, Embedding

logger = structlog.get_logger()


class OllamaEmbedder:
    """Embeds text chunks via the Ollama /api/embeddings endpoint."""

    def __init__(self, model: str, base_url: str) -> None:
        """Initialize the embedder.

        Args:
            model: Ollama model name (e.g. nomic-embed-text).
            base_url: Base URL of the Ollama server.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")

    async def embed(self, chunks: list[Chunk]) -> list[Embedding]:
        """Send each chunk to Ollama and return Embedding objects.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List of Embedding objects with vectors from Ollama.
        """
        embeddings = []

        # Embed each chunk sequentially via Ollama REST API
        async with httpx.AsyncClient() as client:
            for chunk in chunks:
                response = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": chunk.content},
                    timeout=60.0,
                )
                response.raise_for_status()
                vector: list[float] = response.json()["embedding"]
                embeddings.append(
                    Embedding(chunk=chunk, vector=vector, model=self._model)
                )

        logger.info("chunks_embedded", count=len(embeddings), model=self._model)
        return embeddings
