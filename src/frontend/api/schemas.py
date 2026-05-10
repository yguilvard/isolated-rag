from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, SecretStr


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"


class UserClaims(BaseModel):
    """Decoded JWT claims for the authenticated user."""

    user_id: UUID
    is_admin: bool


class CreateUserRequest(BaseModel):
    """Request body for admin user creation."""

    username: str
    password: SecretStr
    is_admin: bool = False


class UserResponse(BaseModel):
    """Response body for a created user (no password hash)."""

    id: UUID
    username: str
    is_admin: bool
    created_at: datetime


class IngestResponse(BaseModel):
    """Response body for a successful ingestion."""

    status: str
    chunks_ingested: int
    document: str


class InfoResponse(BaseModel):
    """Application metadata exposed to the frontend."""

    version: str
    embedding_model: str


class DocumentChunk(BaseModel):
    """A single text chunk belonging to an ingested document."""

    index: int
    content: str


class DocumentSummary(BaseModel):
    """Summary of an ingested document visible to the current user."""

    name: str
    chunks: int
    visibility: str
    uploaded_at: datetime
    avg_chars: int = 0


class SearchResultItem(BaseModel):
    """A single chunk returned by a similarity search."""

    content: str
    document: str
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    """Similarity search results."""

    query: str
    scope: str
    results: list[SearchResultItem]
