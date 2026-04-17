from collections.abc import AsyncGenerator
from typing import Annotated

import asyncpg
import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from src.core.config import Settings
from src.core.rag.chunkers.sentence import SentenceChunker
from src.core.rag.embedders.ollama import OllamaEmbedder
from src.core.rag.store.pgvector import PgVectorStore
from src.core.services.auth import AuthService
from src.core.services.ingest import IngestService
from src.frontend.api.schemas import UserClaims

logger = structlog.get_logger()

_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_settings() -> Settings:
    """Load application settings from YAML + env vars."""
    return Settings.from_yaml()


async def get_db_conn(
    request: Request,
) -> AsyncGenerator[asyncpg.Connection, None]:
    """Yield an asyncpg connection from the shared pool.

    Args:
        request: Current FastAPI request (provides access to app.state.pool).
    """
    # Acquire connection from the lifespan-managed pool and release on exit
    async with request.app.state.pool.acquire() as conn:
        yield conn


async def get_current_user(
    token: Annotated[str, Depends(_oauth2)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UserClaims:
    """Decode the Bearer JWT and return user claims.

    Args:
        token: Bearer token from the Authorization header.
        settings: Application settings for JWT verification.

    Raises:
        HTTPException: 401 if the token is missing, invalid, or expired.
    """
    try:
        # Decode and verify Bearer JWT signature and expiry
        payload = jwt.decode(
            token,
            settings.api.secret_key.get_secret_value(),
            algorithms=[settings.api.algorithm],
        )
        # Extract user claims from validated payload
        return UserClaims(user_id=payload["sub"], is_admin=payload["is_admin"])
    except jwt.InvalidTokenError as exc:
        logger.warning("invalid_token", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def require_admin(
    user: Annotated[UserClaims, Depends(get_current_user)],
) -> UserClaims:
    """Require that the current user has admin privileges.

    Args:
        user: Claims from the validated JWT.

    Raises:
        HTTPException: 403 if the user is not an admin.
    """
    # Reject non-admin callers before any state change
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def get_auth_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AuthService:
    """Create an AuthService from current settings.

    Args:
        settings: Application settings containing API JWT config.

    Returns:
        AuthService configured with JWT signing key and algorithm.
    """
    return AuthService(settings.api)


def get_ingest_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestService:
    """Create an IngestService wired with pipeline components from settings.

    PgVectorStore is lightweight at construction (no connection opened until
    save() is called). The actual DB connection is passed per-request via
    IngestService.run(conn=...), so this factory is safe to call per request.

    Args:
        settings: Application settings for ingestion and database config.

    Returns:
        IngestService with chunker, embedder, and vector store wired from config.
    """
    # Wire pipeline components from configuration
    chunker = SentenceChunker(chunk_sentences=settings.ingestion.chunk_sentences)
    embedder = OllamaEmbedder(
        model=settings.ingestion.embedding_model,
        base_url=settings.ingestion.ollama_url,
    )
    store = PgVectorStore(dsn=settings.database.dsn)
    return IngestService(chunker=chunker, embedder=embedder, store=store)
