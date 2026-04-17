from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from src.frontend.api.app import create_app
from src.frontend.api.deps import get_auth_service, get_current_user, get_db_conn, get_ingest_service  # noqa: F401
from src.frontend.api.schemas import UserClaims


@asynccontextmanager
async def _noop_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
def mock_conn():
    conn = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = txn
    return conn


@pytest.fixture
def admin_claims():
    return UserClaims(user_id=uuid4(), is_admin=True)


@pytest.fixture
def user_claims():
    return UserClaims(user_id=uuid4(), is_admin=False)


@pytest.fixture
def app(mock_conn):
    """FastAPI app with no-op lifespan and mocked DB connection."""
    _app = create_app(lifespan=_noop_lifespan)

    async def override_db_conn() -> AsyncGenerator:
        yield mock_conn

    _app.dependency_overrides[get_db_conn] = override_db_conn
    return _app


@pytest.fixture
async def client(app):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
