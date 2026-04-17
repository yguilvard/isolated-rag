import asyncio
from pathlib import Path

import structlog
import typer

from src.core.config import Settings
from src.core.rag.chunkers.sentence import SentenceChunker
from src.core.rag.embedders.ollama import OllamaEmbedder
from src.core.rag.loaders.pdf import PDFLoader
from src.core.rag.loaders.text import TextLoader
from src.core.rag.pipeline import IngestionPipeline
from src.core.rag.store.pgvector import PgVectorStore

app = typer.Typer(help="isolated-rag CLI")
logger = structlog.get_logger()

_LOADER_MAP = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


@app.command()
def ingest(path: Path = typer.Argument(..., help="Path to document to ingest")) -> None:
    """Ingest a document into the RAG vector store."""
    # Validate supported file type
    if path.suffix not in _LOADER_MAP:
        raise typer.BadParameter(
            f"Unsupported file type {path.suffix!r}. Supported: .pdf, .txt, .md"
        )

    # Load settings from config/base.yaml + env vars
    settings = Settings.from_yaml()

    # Wire up pipeline components
    loader = _LOADER_MAP[path.suffix]()
    chunker = SentenceChunker(chunk_sentences=settings.ingestion.chunk_sentences)
    embedder = OllamaEmbedder(
        model=settings.ingestion.embedding_model,
        base_url=settings.ingestion.ollama_url,
    )
    store = PgVectorStore(dsn=settings.database.dsn)
    pipeline = IngestionPipeline(loader, chunker, embedder, store)

    async def _run() -> None:
        await store.setup()
        await pipeline.run(path)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
