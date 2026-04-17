import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Callable

import asyncpg
import structlog
from fastapi import FastAPI

from src.core.config import Settings
from src.core.rag.store.pgvector import PgVectorStore
from src.core.services.auth import AuthService

logger = structlog.get_logger()

# SQL run at startup to create the users table and RLS policies.
# All statements are idempotent.
_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE constraint_name = 'embeddings_owner_id_fkey'
      AND table_name = 'embeddings'
  ) THEN
    ALTER TABLE embeddings
      ADD CONSTRAINT embeddings_owner_id_fkey
      FOREIGN KEY (owner_id) REFERENCES users(id);
  END IF;
END $$;

ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings FORCE ROW LEVEL SECURITY;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE policyname = 'embeddings_select' AND tablename = 'embeddings'
  ) THEN
    CREATE POLICY embeddings_select ON embeddings FOR SELECT
      USING (
        visibility = 'public'
        OR owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID
      );
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE policyname = 'embeddings_insert' AND tablename = 'embeddings'
  ) THEN
    CREATE POLICY embeddings_insert ON embeddings FOR INSERT
      WITH CHECK (
        owner_id = nullif(current_setting('app.current_user_id', true), '')::UUID
      );
  END IF;
END $$;
"""


async def _run_migrations(conn: asyncpg.Connection, settings: Settings) -> None:
    """Run DB migrations: base table, users table, FK, RLS policies, admin seed.

    Args:
        conn: asyncpg connection for executing migration SQL.
        settings: Application settings used to build the PgVectorStore DSN.
    """
    # Ensure vector extension + embeddings table + owner columns exist
    store = PgVectorStore(dsn=settings.database.dsn)
    await store.setup()

    # Create users table, FK, RLS policies
    await conn.execute(_MIGRATION_SQL)
    logger.info("migrations_complete")

    # Seed first admin from env vars if no admin exists
    admin_user = os.environ.get("ADMIN_USERNAME")
    admin_pass = os.environ.get("ADMIN_PASSWORD")
    if admin_user and admin_pass:
        existing = await conn.fetchval("SELECT COUNT(*) FROM users WHERE is_admin = TRUE")
        if existing == 0:
            auth = AuthService(settings.api)
            await auth.create_user(conn, username=admin_user, password=admin_pass, is_admin=True)
            logger.info("admin_seeded", username=admin_user)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create pool, run migrations, seed admin on startup; close pool on shutdown."""
    settings = Settings.from_yaml()
    # Reject empty signing key before accepting any traffic
    if not settings.api.secret_key.get_secret_value():
        raise RuntimeError("api.secret_key must be set (use API__SECRET_KEY env var)")
    # Create the shared asyncpg connection pool
    app.state.pool = await asyncpg.create_pool(settings.database.dsn)
    async with app.state.pool.acquire() as conn:
        await _run_migrations(conn, settings)
    logger.info("api_ready")
    yield
    # Drain and close the pool on shutdown
    await app.state.pool.close()
    logger.info("api_shutdown")


def create_app(lifespan: Callable | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        lifespan: Optional lifespan context manager override (used in tests).

    Returns:
        Configured FastAPI application instance.
    """
    import importlib.metadata
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles
    from src.frontend.api.routers import auth as auth_router
    from src.frontend.api.routers import ingest as ingest_router
    from src.frontend.api.schemas import InfoResponse

    # Use the provided lifespan or the production one
    actual_lifespan = lifespan if lifespan is not None else _lifespan
    app = FastAPI(title="isolated-rag API", lifespan=actual_lifespan)

    @app.get("/info", response_model=InfoResponse)
    def get_info() -> InfoResponse:
        """Return application version and active embedding model."""
        settings = Settings.from_yaml()
        return InfoResponse(
            version=importlib.metadata.version("isolated-rag"),
            embedding_model=settings.ingestion.embedding_model,
        )

    # Register API routers before static files so API routes take precedence
    app.include_router(auth_router.router)
    app.include_router(ingest_router.router)

    # Serve the built SPA — only if dist/ exists (skipped in Vite dev mode)
    _web_dist = Path(__file__).parent.parent / "web" / "dist"
    if _web_dist.exists():
        app.mount("/", StaticFiles(directory=_web_dist, html=True), name="spa")

    return app
