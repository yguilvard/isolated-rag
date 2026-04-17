import tempfile
from pathlib import Path
from typing import Annotated, Literal

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from src.core.services.ingest import IngestService
from src.frontend.api.deps import get_current_user, get_db_conn, get_ingest_service
from src.frontend.api.schemas import IngestResponse, UserClaims

logger = structlog.get_logger()

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    user: Annotated[UserClaims, Depends(get_current_user)],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    service: Annotated[IngestService, Depends(get_ingest_service)],
    file: UploadFile,
    visibility: Annotated[Literal["private", "public"], Form()] = "private",
) -> IngestResponse:
    """Ingest an uploaded document into the user's vector store.

    Saves the upload to a temp file, runs the ingestion pipeline with the
    caller's user ID and chosen visibility, then removes the temp file.

    Args:
        user: Authenticated user claims from JWT.
        conn: asyncpg connection (used for RLS inside IngestService).
        service: IngestService wired with pipeline components.
        file: Uploaded document (.pdf, .txt, or .md).
        visibility: "private" (default) or "public".

    Returns:
        IngestResponse with chunk count and document name.

    Raises:
        HTTPException: 422 for unsupported file types.
        HTTPException: 503 if the embedding service is unavailable.
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    # Write upload to a temp file so existing loaders can read from Path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    tmp_path.write_bytes(await file.read())

    try:
        # Run the ingestion pipeline with RLS context
        chunks_ingested = await service.run(
            tmp_path,
            user_id=user.user_id,
            visibility=visibility,
            conn=conn,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except ConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    logger.info("ingest_request_complete", filename=filename, chunks=chunks_ingested)
    return IngestResponse(
        status="ok",
        chunks_ingested=chunks_ingested,
        document=filename,
    )
