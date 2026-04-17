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
