from datetime import datetime, timedelta, timezone
from uuid import UUID

import asyncpg
import bcrypt
import jwt
import structlog

from src.core.config import ApiSettings

logger = structlog.get_logger()


class AuthService:
    """Handles password hashing, JWT issuance/verification, and user management."""

    def __init__(self, settings: ApiSettings) -> None:
        """Initialize with API settings.

        Args:
            settings: API configuration including secret key and JWT parameters.
        """
        self._settings = settings

    def hash_password(self, password: str) -> str:
        """Hash a plain-text password with bcrypt.

        Args:
            password: Plain-text password to hash.

        Returns:
            bcrypt hash string suitable for DB storage.
        """
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Check a plain-text password against a stored bcrypt hash.

        Args:
            password: Plain-text candidate password.
            hashed: Stored bcrypt hash string.

        Returns:
            True if the password matches, False otherwise.
        """
        return bcrypt.checkpw(password.encode(), hashed.encode())

    def create_token(self, user_id: UUID, *, is_admin: bool) -> str:
        """Create a signed JWT for the given user.

        Args:
            user_id: User UUID to embed in the token subject.
            is_admin: Whether the user has admin privileges.

        Returns:
            Signed JWT string.
        """
        # Build JWT payload with subject, admin flag, and expiry
        payload = {
            "sub": str(user_id),
            "is_admin": is_admin,
            "exp": datetime.now(timezone.utc)
            + timedelta(minutes=self._settings.token_expire_minutes),
        }
        return jwt.encode(
            payload,
            self._settings.secret_key.get_secret_value(),
            algorithm=self._settings.algorithm,
        )

    def decode_token(self, token: str) -> dict:
        """Decode and verify a JWT.

        Args:
            token: Signed JWT string.

        Returns:
            Decoded payload dict.

        Raises:
            jwt.InvalidTokenError: If the token is invalid or expired.
        """
        # Verify signature and expiry, return payload
        return jwt.decode(
            token,
            self._settings.secret_key.get_secret_value(),
            algorithms=[self._settings.algorithm],
        )

    async def create_user(
        self,
        conn: asyncpg.Connection,
        *,
        username: str,
        password: str,
        is_admin: bool = False,
    ) -> dict:
        """Insert a new user into the database.

        Args:
            conn: asyncpg connection.
            username: Unique username.
            password: Plain-text password (hashed before storage).
            is_admin: Grant admin privileges.

        Returns:
            Dict of the created user record (id, username, is_admin, created_at).
        """
        # Hash password before storage
        password_hash = self.hash_password(password)
        row = await conn.fetchrow(
            "INSERT INTO users (username, password_hash, is_admin)"
            " VALUES ($1, $2, $3)"
            " RETURNING id, username, is_admin, created_at",
            username,
            password_hash,
            is_admin,
        )
        logger.info("user_created", username=username, is_admin=is_admin)
        return dict(row)

    async def login(self, conn: asyncpg.Connection, *, username: str, password: str) -> str:
        """Validate credentials and return a signed JWT.

        Args:
            conn: asyncpg connection.
            username: Username to look up.
            password: Plain-text password to verify.

        Returns:
            Signed JWT string.

        Raises:
            ValueError: If credentials are invalid.
        """
        # Fetch user record by username
        row = await conn.fetchrow(
            "SELECT id, password_hash, is_admin FROM users WHERE username = $1",
            username,
        )
        if row is None or not self.verify_password(password, row["password_hash"]):
            raise ValueError("Invalid credentials")
        token = self.create_token(UUID(str(row["id"])), is_admin=row["is_admin"])
        logger.info("user_login", username=username)
        return token
