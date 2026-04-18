from pathlib import Path
from typing import Annotated, Literal

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.frontend.api.deps import get_current_user, get_db_conn
from src.frontend.api.schemas import DocumentChunk, DocumentSummary, UserClaims

logger = structlog.get_logger()

router = APIRouter(tags=["documents"])

_LIST_SQL = """
    SELECT
        document_path                       AS name,
        COUNT(*)::int                       AS chunks,
        visibility,
        MIN(created_at)                     AS uploaded_at,
        AVG(char_length(content))::int      AS avg_chars
    FROM embeddings
    GROUP BY document_path, visibility
    ORDER BY MIN(created_at) DESC
"""

_CHUNKS_SQL = """
    SELECT chunk_index, content
    FROM embeddings
    WHERE document_path = $1
       OR document_path LIKE '%/' || $1
    ORDER BY chunk_index
"""

_DELETE_SQL = """
    DELETE FROM embeddings
    WHERE (document_path = $1 OR document_path LIKE '%/' || $1)
      AND owner_id = $2
"""

_SHARE_SQL = """
    UPDATE embeddings
    SET visibility = $2
    WHERE (document_path = $1 OR document_path LIKE '%/' || $1)
      AND owner_id = $3
"""


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
) -> list[DocumentSummary]:
    """Return all documents accessible to the current user.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection with RLS context set.

    Returns:
        List of DocumentSummary objects, newest first.
    """
    rows = await conn.fetch(_LIST_SQL)
    return [
        DocumentSummary(
            name=Path(row["name"]).name,
            chunks=row["chunks"],
            visibility=row["visibility"],
            uploaded_at=row["uploaded_at"],
            avg_chars=row["avg_chars"] or 0,
        )
        for row in rows
    ]


@router.get("/documents/chunks", response_model=list[DocumentChunk])
async def get_chunks(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    name: Annotated[str, Query(min_length=1)],
) -> list[DocumentChunk]:
    """Return the text chunks stored for a given document.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection with RLS context set.
        name: Exact document name as stored (from the document list).

    Returns:
        Ordered list of DocumentChunk objects.
    """
    rows = await conn.fetch(_CHUNKS_SQL, name)
    return [DocumentChunk(index=row["chunk_index"], content=row["content"]) for row in rows]


@router.delete("/documents", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    name: Annotated[str, Query(min_length=1)],
) -> None:
    """Delete all chunks of a document owned by the current user.

    Only the document owner can delete it; shared documents owned by others
    are unaffected (RLS WHERE clause on owner_id ensures this).

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection with RLS context set.
        name: Exact document name as returned by the document list.

    Raises:
        HTTPException: 404 if no owned rows matched the given name.
    """
    result = await conn.execute(_DELETE_SQL, name, user.user_id)
    # result is e.g. "DELETE 5" — extract row count
    deleted = int(result.split()[-1])
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or not owned by you.",
        )
    logger.info("document_deleted", name=name, chunks=deleted, user_id=str(user.user_id))


@router.patch("/documents/visibility", status_code=status.HTTP_204_NO_CONTENT)
async def set_visibility(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    name: Annotated[str, Query(min_length=1)],
    visibility: Annotated[Literal["private", "public"], Query()],
) -> None:
    """Change the visibility of a document owned by the current user.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection.
        name: Exact document name as returned by the document list.
        visibility: Target visibility — "private" or "public".

    Raises:
        HTTPException: 404 if no owned rows matched the given name.
    """
    result = await conn.execute(_SHARE_SQL, name, visibility, user.user_id)
    updated = int(result.split()[-1])
    if updated == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or not owned by you.",
        )
    logger.info("visibility_changed", name=name, visibility=visibility, user_id=str(user.user_id))
