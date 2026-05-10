import pytest
from unittest.mock import AsyncMock, MagicMock

from src.frontend.api.deps import get_current_user, get_ingest_service


@pytest.fixture
def mock_ingest_service():
    svc = MagicMock()
    svc.run = AsyncMock(return_value=7)
    return svc


@pytest.fixture
def authed_app(app, user_claims, mock_ingest_service):
    """App with mocked auth, ingest service, and DB connection (from conftest `app`)."""
    app.dependency_overrides[get_current_user] = lambda: user_claims
    app.dependency_overrides[get_ingest_service] = lambda: mock_ingest_service
    return app


async def test_ingest_returns_200_with_chunk_count(authed_app, client):
    response = await client.post(
        "/ingest",
        files={"file": ("doc.txt", b"Hello world. This is a test.", "text/plain")},
        data={"visibility": "private"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks_ingested"] == 7
    assert body["document"] == "doc.txt"


async def test_ingest_defaults_visibility_to_private(authed_app, client, mock_ingest_service):
    await client.post(
        "/ingest",
        files={"file": ("note.md", b"# Note", "text/markdown")},
    )
    assert mock_ingest_service.run.call_args.kwargs["visibility"] == "private"


async def test_ingest_requires_authentication(app, client):
    # No get_current_user override — uses real dep which requires a token
    response = await client.post(
        "/ingest",
        files={"file": ("doc.txt", b"Hello", "text/plain")},
    )
    assert response.status_code == 401


async def test_ingest_passes_user_id_to_service(authed_app, client, mock_ingest_service, user_claims):
    await client.post(
        "/ingest",
        files={"file": ("doc.txt", b"Hello world.", "text/plain")},
        data={"visibility": "public"},
    )
    assert mock_ingest_service.run.call_args.kwargs["user_id"] == user_claims.user_id
