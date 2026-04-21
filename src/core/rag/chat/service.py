from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import asyncpg
import structlog
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings

logger = structlog.get_logger()

_PROMPTS_DIR = Path(__file__).parent / "prompts"

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


@dataclass
class RetrievedChunk:
    """A chunk returned by vector similarity search."""

    content: str
    document: str
    chunk_index: int
    score: float


class ChatService:
    """Handles RAG retrieval and streaming LLM generation.

    Retrieval uses the same pgvector similarity search as the search
    endpoint.  Generation streams tokens from a ``ChatOllama`` instance
    using role-separated prompts loaded from Jinja2 templates.
    """

    def __init__(self, embedding_model: str, embedding_url: str) -> None:
        """Initialise the service with the embedding model used for retrieval.

        Args:
            embedding_model: Ollama model name for query embedding.
            embedding_url: Base URL of the embedding Ollama server.
        """
        self._embedder = OllamaEmbeddings(
            model=embedding_model, base_url=embedding_url
        )
        env = Environment(loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False)
        self._rag_system = env.get_template("rag_system.j2")
        self._rag_user = env.get_template("rag_user.j2")
        self._plain_system = env.get_template("plain_system.j2")

    async def retrieve(
        self,
        query: str,
        conn: asyncpg.Connection,
        user_id: UUID,
        scope: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Embed *query* and return the top-K most relevant chunks.

        Args:
            query: User question to embed and search for.
            conn: asyncpg connection with RLS already activated.
            user_id: UUID of the authenticated caller (for RLS).
            scope: Visibility filter — ``"private"``, ``"public"``, or ``"all"``.
            top_k: Maximum number of chunks to return.

        Returns:
            Ordered list of chunks, most relevant first.
        """
        vector: list[float] = await self._embedder.aembed_query(query)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)", str(user_id)
            )
            rows = await conn.fetch(_SEARCH_SQL, vector_str, scope, top_k)

        chunks = [
            RetrievedChunk(
                content=row["content"],
                document=row["document_path"],
                chunk_index=row["chunk_index"],
                score=float(row["score"]),
            )
            for row in rows
        ]
        logger.info("rag_retrieved", query=query[:80], scope=scope, hits=len(chunks))
        return chunks

    async def stream(
        self,
        query: str,
        llm: ChatOllama,
        use_rag: bool,
        chunks: list[RetrievedChunk],
    ) -> AsyncGenerator[str, None]:
        """Stream answer tokens from the LLM.

        When *use_rag* is True, the system prompt instructs the model to
        answer strictly from the provided context.  When False, it uses
        plain training knowledge with an explicit honesty instruction.

        Args:
            query: The user question.
            llm: Pre-configured ``ChatOllama`` instance for the chosen model.
            use_rag: Whether to inject retrieved context into the prompt.
            chunks: Retrieved chunks (used only when *use_rag* is True).

        Yields:
            Raw text tokens as they arrive from the model.
        """
        if use_rag:
            system_content = self._rag_system.render()
            user_content = self._rag_user.render(chunks=chunks, query=query)
        else:
            system_content = self._plain_system.render()
            user_content = query

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        async for token in llm.astream(messages):
            if token.content:
                yield str(token.content)
