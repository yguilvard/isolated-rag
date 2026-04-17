import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import jwt
from pydantic import SecretStr

from src.core.config import ApiSettings
from src.core.services.auth import AuthService


@pytest.fixture
def settings() -> ApiSettings:
    return ApiSettings(
        secret_key=SecretStr("test-secret-key-long-enough-xxxxx"),  # 32+ bytes for HS256
        algorithm="HS256",
        token_expire_minutes=60,
    )


@pytest.fixture
def service(settings: ApiSettings) -> AuthService:
    return AuthService(settings)


def test_hash_password_returns_bcrypt_hash(service: AuthService) -> None:
    hashed = service.hash_password("mypassword")
    assert hashed.startswith("$2b$")


def test_verify_password_correct(service: AuthService) -> None:
    hashed = service.hash_password("mypassword")
    assert service.verify_password("mypassword", hashed) is True


def test_verify_password_wrong(service: AuthService) -> None:
    hashed = service.hash_password("mypassword")
    assert service.verify_password("wrong", hashed) is False


def test_create_token_decodes_to_correct_claims(service: AuthService, settings: ApiSettings) -> None:
    uid = uuid4()
    token = service.create_token(uid, is_admin=True)
    payload = jwt.decode(
        token,
        settings.secret_key.get_secret_value(),
        algorithms=[settings.algorithm],
    )
    assert payload["sub"] == str(uid)
    assert payload["is_admin"] is True
    assert payload["exp"] > datetime.now(timezone.utc).timestamp()


def test_decode_token_returns_payload(service: AuthService) -> None:
    uid = uuid4()
    token = service.create_token(uid, is_admin=False)
    payload = service.decode_token(token)
    assert payload["sub"] == str(uid)
    assert payload["is_admin"] is False


@pytest.mark.asyncio
async def test_create_user_calls_insert_and_returns_record(service: AuthService) -> None:
    uid = uuid4()
    conn = AsyncMock()
    conn.fetchrow.return_value = {
        "id": uid,
        "username": "alice",
        "is_admin": False,
        "created_at": datetime.now(timezone.utc),
    }
    result = await service.create_user(conn, username="alice", password="secret", is_admin=False)
    assert result["username"] == "alice"
    assert result["id"] == uid
    conn.fetchrow.assert_called_once()
    # Verify the INSERT SQL is called, not raw password
    sql = conn.fetchrow.call_args.args[0]
    assert "INSERT INTO users" in sql


@pytest.mark.asyncio
async def test_login_returns_token_on_valid_credentials(service: AuthService) -> None:
    uid = uuid4()
    conn = AsyncMock()
    hashed = service.hash_password("secret")
    conn.fetchrow.return_value = {"id": uid, "password_hash": hashed, "is_admin": False}
    token = await service.login(conn, username="alice", password="secret")
    assert isinstance(token, str)
    payload = service.decode_token(token)
    assert payload["sub"] == str(uid)


@pytest.mark.asyncio
async def test_login_raises_on_wrong_password(service: AuthService) -> None:
    conn = AsyncMock()
    hashed = service.hash_password("correct")
    conn.fetchrow.return_value = {"id": uuid4(), "password_hash": hashed, "is_admin": False}
    with pytest.raises(ValueError, match="Invalid credentials"):
        await service.login(conn, username="alice", password="wrong")


@pytest.mark.asyncio
async def test_login_raises_when_user_not_found(service: AuthService) -> None:
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    with pytest.raises(ValueError, match="Invalid credentials"):
        await service.login(conn, username="ghost", password="any")


@pytest.mark.asyncio
async def test_create_user_raises_when_insert_returns_none(service: AuthService) -> None:
    conn = AsyncMock()
    conn.fetchrow.return_value = None
    with pytest.raises(RuntimeError, match="INSERT INTO users returned no record"):
        await service.create_user(conn, username="alice", password="secret")
