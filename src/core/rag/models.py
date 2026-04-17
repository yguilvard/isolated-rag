from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class Document(BaseModel):
    """A loaded document before chunking."""

    path: Path
    content: str
    mime_type: str  # "application/pdf" | "text/plain" | "text/markdown"


class Chunk(BaseModel):
    """A sentence-grouped chunk of a document."""

    document_path: Path
    index: int
    content: str


class Embedding(BaseModel):
    """A chunk paired with its embedding vector and ownership metadata."""

    chunk: Chunk
    vector: list[float]
    model: str
    owner_id: UUID | None = None
    visibility: Literal["private", "public"] = "private"
