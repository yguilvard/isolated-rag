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

# Hybrid search: Reciprocal Rank Fusion of vector similarity + BM25 full-text.
#
# vs  — top-60 chunks by cosine distance, pre-filtered by minimum cosine score ($5)
# fts — top-60 chunks matching the query via PostgreSQL full-text (BM25 branch)
# rrf — LEFT JOIN so BM25 can only boost chunks already in vs; BM25-only results
#        (cosine below threshold) are excluded, preserving the relevance gate.
#
# Parameters: $1=query_vector, $2=query_text (FTS), $3=scope, $4=limit, $5=min_cosine
_SEARCH_SQL = """
WITH vs AS (
    SELECT chunk_index, document_path, content,
           1 - (vector <=> $1::vector) AS cosine_score,
           ROW_NUMBER() OVER (ORDER BY vector <=> $1::vector) AS vrank
    FROM embeddings
    WHERE (
        ($3 = 'all' AND (
            visibility = 'public'
            OR owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID
        ))
        OR ($3 = 'private'
            AND visibility = 'private'
            AND owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID)
        OR ($3 = 'public' AND visibility = 'public')
    )
    AND 1 - (vector <=> $1::vector) >= $5
    ORDER BY vector <=> $1::vector
    LIMIT 60
),
fts AS (
    SELECT chunk_index, document_path, content,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(to_tsvector('french', content),
                                websearch_to_tsquery('french', $2)) DESC
           ) AS frank
    FROM embeddings
    WHERE (
        ($3 = 'all' AND (
            visibility = 'public'
            OR owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID
        ))
        OR ($3 = 'private'
            AND visibility = 'private'
            AND owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID)
        OR ($3 = 'public' AND visibility = 'public')
    )
    AND to_tsvector('french', content) @@ websearch_to_tsquery('french', $2)
    LIMIT 60
),
rrf AS (
    SELECT
        vs.chunk_index,
        vs.document_path,
        vs.content,
        vs.cosine_score,
        1.0 / (60 + vs.vrank) +
        COALESCE(1.0 / (60 + fts.frank), 0.0) AS rrf_score
    FROM vs
    LEFT JOIN fts
        ON vs.chunk_index = fts.chunk_index
       AND vs.document_path = fts.document_path
)
SELECT chunk_index, document_path, content, cosine_score AS score
FROM rrf
ORDER BY rrf_score DESC
LIMIT $4
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

    Retrieval uses Reciprocal Rank Fusion (RRF) over:
    - pgvector cosine similarity on HyDE-expanded query embeddings
    - PostgreSQL full-text search on the original query text

    Generation streams tokens from a ``ChatOllama`` instance using
    role-separated prompts loaded from Jinja2 templates.
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
        self._hyde_system = env.get_template("hyde_system.j2")
        self._hyde_user = env.get_template("hyde_user.j2")

    async def _expand_query_hyde(self, query: str, llm: ChatOllama) -> str:
        """Generate a hypothetical document passage for HyDE retrieval.

        Embeds the generated passage instead of the raw query to bridge the
        lexical/semantic gap between short queries and document-length content.
        Falls back to the original query on any LLM error.

        Args:
            query: Original user query.
            llm: Chat model to use for generation.

        Returns:
            Hypothetical passage string, or *query* on failure.
        """
        try:
            response = await llm.ainvoke([
                SystemMessage(content=self._hyde_system.render()),
                HumanMessage(content=self._hyde_user.render(query=query)),
            ])
            passage = str(response.content).strip()
            if passage:
                logger.info("hyde_expanded", query=query[:60], passage=passage[:120])
                return passage
        except Exception as exc:
            logger.warning("hyde_expansion_failed", error=str(exc))
        return query

    async def retrieve(
        self,
        query: str,
        conn: asyncpg.Connection,
        user_id: UUID,
        scope: str,
        top_k: int,
        llm: ChatOllama | None = None,
        min_score: float = 0.55,
    ) -> list[RetrievedChunk]:
        """Embed *query* (optionally via HyDE) and return top-K chunks via RRF.

        Only chunks whose cosine similarity meets *min_score* enter the vector
        branch.  When no chunk clears the threshold the returned list is empty,
        signalling to the caller that no relevant context was found.

        HyDE is skipped for queries shorter than three words — short technical
        terms are often ambiguous and the LLM expansion drifts to unrelated
        interpretations, worsening retrieval.

        Args:
            query: User question to search for.
            conn: asyncpg connection with RLS already activated.
            user_id: UUID of the authenticated caller (for RLS).
            scope: Visibility filter — ``"private"``, ``"public"``, or ``"all"``.
            top_k: Maximum number of chunks to return.
            llm: When provided and the query has ≥ 3 words, HyDE is applied.
                The original query text is still used for the FTS branch.
            min_score: Minimum cosine similarity for a chunk to enter ranking.

        Returns:
            Ordered list of chunks, highest RRF score first.
            Empty when no chunk meets *min_score*.
        """
        # HyDE only helps multi-word natural-language queries; skip for short
        # technical terms where the LLM expansion tends to drift semantically.
        use_hyde = llm is not None and len(query.split()) >= 3
        embed_text = await self._expand_query_hyde(query, llm) if use_hyde else query
        vector: list[float] = await self._embedder.aembed_query(embed_text)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        async with conn.transaction():
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, true)", str(user_id)
            )
            rows = await conn.fetch(_SEARCH_SQL, vector_str, query, scope, top_k, min_score)

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
        if use_rag and chunks:
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
