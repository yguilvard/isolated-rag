from typing import Annotated, Literal

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Form
from langchain_ollama import OllamaEmbeddings

from src.core.config import Settings
from src.frontend.api.deps import get_current_user, get_db_conn, get_settings
from src.frontend.api.schemas import SearchResponse, SearchResultItem, UserClaims

logger = structlog.get_logger()

router = APIRouter(tags=["search"])

_SEARCH_SQL = """
    SELECT content, document_path, chunk_index,
           1 - (vector <=> $1::vector) AS score
    FROM embeddings
    WHERE (
        ($2 = 'all' AND (
            visibility = 'public'
            OR owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID
        ))
        OR ($2 = 'private'
            AND visibility = 'private'
            AND owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID)
        OR ($2 = 'public' AND visibility = 'public')
    )
    ORDER BY vector <=> $1::vector
    LIMIT $3
"""


@router.post("/search", response_model=SearchResponse)
async def search(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    settings: Annotated[Settings, Depends(get_settings)],
    query: Annotated[str, Form()],
    scope: Annotated[Literal["private", "public", "all"], Form()] = "all",
    top_k: Annotated[int, Form(ge=1, le=50)] = 5,
) -> SearchResponse:
    """Embed a query and return the top-K most similar chunks.

    RLS is activated so only chunks the caller is allowed to see are returned.
    The `scope` parameter further narrows results to private-only, public-only,
    or all accessible chunks.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection with RLS context set.
        settings: Application settings for embedding model config.
        query: Free-text search query.
        scope: Document visibility filter — "private", "public", or "all".
        top_k: Maximum number of results to return (1-50).

    Returns:
        SearchResponse with ranked results and echoed query/scope.
    """
    # Embed the query string using the active embedding model
    lc = OllamaEmbeddings(
        model=settings.ingestion.embedding_model,
        base_url=settings.ingestion.ollama_url,
    )
    vector: list[float] = await lc.aembed_query(query)
    vector_str = "[" + ",".join(str(v) for v in vector) + "]"

    # Set RLS session variable so row-level policies can filter by owner
    async with conn.transaction():
        await conn.execute(
            "SELECT set_config('app.current_user_id', $1, true)", str(user.user_id)
        )
        rows = await conn.fetch(_SEARCH_SQL, vector_str, scope, top_k)

    results = [
        SearchResultItem(
            content=row["content"],
            document=row["document_path"],
            chunk_index=row["chunk_index"],
            score=float(row["score"]),
        )
        for row in rows
    ]

    logger.info(
        "search_complete",
        query=query[:80],
        scope=scope,
        top_k=top_k,
        hits=len(results),
    )
    return SearchResponse(query=query, scope=scope, results=results)
