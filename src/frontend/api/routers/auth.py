from typing import Annotated

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from src.core.services.auth import AuthService
from src.frontend.api.deps import (
    get_auth_service,
    get_db_conn,
    require_admin,
)
from src.frontend.api.schemas import CreateUserRequest, TokenResponse, UserClaims, UserResponse

logger = structlog.get_logger()

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    auth: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """Authenticate with username and password, return a JWT.

    Args:
        form: OAuth2 form with username and password fields.
        conn: asyncpg connection for user lookup.
        auth: AuthService for credential verification.

    Returns:
        TokenResponse with signed JWT.

    Raises:
        HTTPException: 401 if credentials are invalid.
    """
    try:
        # Verify credentials and issue JWT
        token = await auth.login(conn, username=form.username, password=form.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return TokenResponse(access_token=token)


@router.post("/admin/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: CreateUserRequest,
    conn: Annotated[asyncpg.Connection, Depends(get_db_conn)],
    auth: Annotated[AuthService, Depends(get_auth_service)],
    _: Annotated[UserClaims, Depends(require_admin)],
) -> UserResponse:
    """Create a new user account. Requires admin privileges.

    Args:
        body: CreateUserRequest with username, password, and optional is_admin flag.
        conn: asyncpg connection for user insertion.
        auth: AuthService for user creation.
        _: Admin guard dependency (raises 403 if not admin).

    Returns:
        UserResponse with the created user record (no password hash).
    """
    # Create user with hashed password via AuthService
    record = await auth.create_user(
        conn,
        username=body.username,
        password=body.password.get_secret_value(),
        is_admin=body.is_admin,
    )
    return UserResponse(**record)
