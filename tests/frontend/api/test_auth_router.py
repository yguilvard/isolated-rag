import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from src.frontend.api.deps import get_auth_service, get_current_user


@pytest.fixture
def mock_auth_service():
    return MagicMock()


@pytest.fixture
def app_with_auth(app, mock_auth_service):
    app.dependency_overrides[get_auth_service] = lambda: mock_auth_service
    return app


async def test_login_returns_token_on_valid_credentials(app_with_auth, mock_auth_service, client):
    mock_auth_service.login = AsyncMock(return_value="test.jwt.token")
    response = await client.post(
        "/auth/login",
        data={"username": "alice", "password": "secret"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["access_token"] == "test.jwt.token"
    assert body["token_type"] == "bearer"


async def test_login_returns_401_on_invalid_credentials(app_with_auth, mock_auth_service, client):
    mock_auth_service.login = AsyncMock(side_effect=ValueError("Invalid credentials"))
    response = await client.post(
        "/auth/login",
        data={"username": "alice", "password": "wrong"},
    )
    assert response.status_code == 401


async def test_create_user_requires_admin(app, user_claims, client):
    app.dependency_overrides[get_current_user] = lambda: user_claims
    response = await client.post(
        "/admin/users",
        json={"username": "bob", "password": "pw"},
    )
    assert response.status_code == 403


async def test_create_user_as_admin_returns_201(app_with_auth, mock_auth_service, admin_claims, client):
    # Override current_user to return admin; require_admin checks is_admin=True → passes
    app_with_auth.dependency_overrides[get_current_user] = lambda: admin_claims
    uid = uuid4()
    mock_auth_service.create_user = AsyncMock(return_value={
        "id": uid,
        "username": "bob",
        "is_admin": False,
        "created_at": datetime.now(timezone.utc),
    })
    response = await client.post(
        "/admin/users",
        json={"username": "bob", "password": "secure_password"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "bob"
    assert "password" not in body
    assert "password_hash" not in body
