from pathlib import Path

import structlog

from src.core.rag.protocols import Chunker, Embedder, Loader, Store

logger = structlog.get_logger()


class IngestionPipeline:
    """Orchestrates the full document ingestion flow."""

    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedder: Embedder,
        store: Store,
    ) -> None:
        """Initialize the pipeline with its four stage components.

        Args:
            loader: Loads a document from disk.
            chunker: Splits a document into chunks.
            embedder: Embeds chunks into vectors.
            store: Persists embeddings to the vector store.
        """
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store

    async def run(self, path: Path) -> None:
        """Load, chunk, embed and store a single document.

        Args:
            path: Path to the document to ingest.
        """
        logger.info("ingestion_started", path=str(path))

        # Load document from disk
        document = self._loader.load(path)

        # Split into sentence-grouped chunks
        chunks = self._chunker.chunk(document)

        # Embed each chunk via Ollama
        embeddings = await self._embedder.embed(chunks)

        # Persist to pgvector
        await self._store.save(embeddings)

        logger.info("ingestion_complete", path=str(path), chunks=len(chunks))
