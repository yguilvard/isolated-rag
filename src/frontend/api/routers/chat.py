import json
from typing import Annotated, Literal

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_ollama import ChatOllama

from src.core.config import Settings
from src.core.rag.chat.service import ChatService
from src.frontend.api.deps import get_current_user, get_db_conn, get_settings
from src.frontend.api.schemas import UserClaims

logger = structlog.get_logger()

router = APIRouter(tags=["chat"])


def _get_chat_service(settings: Settings) -> ChatService:
    """Instantiate ChatService from application settings."""
    return ChatService(
        embedding_model=settings.ingestion.embedding_model,
        embedding_url=settings.ingestion.ollama_url,
    )


def _sse(event: dict) -> str:
    """Serialise *event* as a single SSE data line."""
    return f"data: {json.dumps(event)}\n\n"


@router.post("/chat")
async def chat(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    settings: Annotated[Settings, Depends(get_settings)],
    query: Annotated[str, Form(min_length=1, max_length=2000)],
    model_id: Annotated[str, Form(max_length=50)],
    use_rag: Annotated[bool, Form()] = True,
    scope: Annotated[Literal["private", "public", "all"], Form()] = "all",
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
) -> StreamingResponse:
    """Stream an LLM answer to *query*, optionally grounded with RAG context.

    The response is a ``text/event-stream`` of JSON events:

    * ``{"type": "sources", "items": [...]}`` — emitted first when RAG is on,
      lists the retrieved chunks used as context.
    * ``{"type": "token", "content": "..."}`` — one event per streamed token.
    * ``{"type": "done"}`` — marks the end of a successful stream.
    * ``{"type": "error", "message": "..."}`` — emitted on failure.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection used for RAG retrieval (RLS activated inside).
        settings: Application settings (chat providers + embedding config).
        query: The user question.
        model_id: ID of the chat provider to use (must be in ``chat_providers``).
        use_rag: When True, retrieved document chunks are injected into the prompt.
        scope: Visibility filter for RAG retrieval.
        top_k: Maximum number of chunks to retrieve (1-20).

    Returns:
        StreamingResponse with ``text/event-stream`` content.

    Raises:
        HTTPException: 422 if *model_id* is not a configured provider.
    """
    # Resolve chat provider from config
    provider = next((p for p in settings.chat_providers if p.id == model_id), None)
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown model: {model_id!r}",
        )

    llm = ChatOllama(
        model=provider.model,
        base_url=provider.ollama_url,
        temperature=0.3,
    )
    service = _get_chat_service(settings)

    async def generate():
        # Retrieve context chunks when RAG is enabled
        chunks = []
        if use_rag:
            try:
                chunks = await service.retrieve(
                    query, conn, user.user_id, scope, top_k
                )
                sources = [
                    {
                        "document": c.document,
                        "chunk_index": c.chunk_index,
                        "score": round(c.score, 4),
                        "content": c.content,
                    }
                    for c in chunks
                ]
                yield _sse({"type": "sources", "items": sources})
            except Exception as exc:
                logger.warning("rag_retrieval_failed", error=str(exc))
                yield _sse({"type": "sources", "items": []})

        # Stream tokens from the LLM
        try:
            async for token in service.stream(query, llm, use_rag, chunks):
                yield _sse({"type": "token", "content": token})
            yield _sse({"type": "done"})
            logger.info(
                "chat_complete",
                model=model_id,
                use_rag=use_rag,
                rag_chunks=len(chunks),
            )
        except Exception as exc:
            logger.error("chat_stream_failed", error=str(exc))
            yield _sse({"type": "error", "message": "The model failed to respond. Is it running?"})

    return StreamingResponse(generate(), media_type="text/event-stream")
