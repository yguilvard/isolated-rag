# isolated-rag

A local RAG (Retrieval-Augmented Generation) system with JWT-authenticated document ingestion and a React web UI. Built on FastAPI, PostgreSQL with pgvector, and Ollama for local embeddings.

## Architecture

```
src/
  core/            # RAG engine: chunking, embeddings, pgvector store
  frontend/
    api/           # FastAPI server (JWT auth, ingest endpoint, static file serving)
    cli.py         # Command-line ingestion tool
    web/           # React + Vite SPA (login + upload UI)
deployment/
  docker/          # Docker Compose for PostgreSQL
config/
  base.yaml        # Default configuration
```

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.14+ | [python.org](https://python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Node.js | 20+ | `brew install node` |
| Docker | latest | [docker.com](https://docker.com) |
| Ollama | latest | `brew install ollama` |

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd isolated-rag
uv sync
```

### 2. Configure environment

Create `.env` in the project root. The Makefile loads it automatically, so all variables are available to every `make` target:

```bash
# PostgreSQL — used by both Docker Compose and the API server
POSTGRES_USER=rag
POSTGRES_PASSWORD=changeme
POSTGRES_DB=rag_db

# API server — required, server refuses to start without a signing key
API__SECRET_KEY=<run: openssl rand -hex 32>

# Seed an admin account on first startup (optional, only runs when no admin exists)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=changeme

# pgAdmin (optional)
PGADMIN_DEFAULT_EMAIL=admin@example.com
PGADMIN_DEFAULT_PASSWORD=admin
```

The Makefile automatically maps `POSTGRES_PASSWORD` → `DATABASE__PASSWORD` for the API server, so a single password variable covers both Docker Compose and the app.

### 3. Start PostgreSQL + Ollama

```bash
make up
```

This starts both PostgreSQL and an Ollama container. The model is pulled automatically the first time you run `make serve` or `make dev`.

---

## Development

Two processes run concurrently: the FastAPI server on port 8000, and the Vite dev server on port 5173. Vite proxies `/auth` and `/ingest` requests to FastAPI — no CORS configuration needed.

```bash
make dev
```

- **Web UI:** http://localhost:5173
- **API docs:** http://localhost:8000/docs

To run them in separate terminals for cleaner logs:

```bash
# Terminal 1 — FastAPI
make serve

# Terminal 2 — Vite dev server
cd src/frontend/web && npm run dev
```

### First login

On startup, if `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set and no admin exists, the server seeds an admin account automatically. Use those credentials to log in at http://localhost:5173.

To create additional users via the API:

```bash
curl -X POST http://localhost:8000/auth/users \
  -H "Authorization: Bearer <admin-token>" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "secret", "is_admin": false}'
```

---

## Production

Build the SPA and serve everything from a single FastAPI process:

```bash
# Build the React app
make build-web

# Start FastAPI (serves both the API and the built SPA)
API__SECRET_KEY="..." ADMIN_USERNAME=admin ADMIN_PASSWORD=changeme make serve
```

- **App:** http://localhost:8000 (serves the SPA)
- **API:** http://localhost:8000/auth/*, http://localhost:8000/ingest

FastAPI mounts the built `dist/` directory only when it exists. In dev mode (no build), the SPA mount is skipped and Vite serves the frontend instead.

---

## CLI ingestion

Ingest a document directly without the web UI:

```bash
make ingest FILE=path/to/document.pdf
```

Supported formats: `.txt`, `.pdf`, `.md`

---

## Configuration

Settings are loaded from `config/base.yaml` and overridden by environment variables using double-underscore separators.

| YAML key | Env var | Default | Description |
|----------|---------|---------|-------------|
| `api.secret_key` | `API__SECRET_KEY` | *(required)* | JWT signing key |
| `api.token_expire_minutes` | `API__TOKEN_EXPIRE_MINUTES` | `60` | JWT lifetime |
| `api.host` | `API__HOST` | `0.0.0.0` | Server bind address |
| `api.port` | `API__PORT` | `8000` | Server port |
| `database.host` | `DATABASE__HOST` | `localhost` | PostgreSQL host |
| `database.port` | `DATABASE__PORT` | `5432` | PostgreSQL port |
| `database.name` | `DATABASE__NAME` | `rag_db` | Database name |
| `database.user` | `DATABASE__USER` | `rag` | Database user |
| `database.password` | `DATABASE__PASSWORD` | *(empty)* | Database password |
| `ingestion.embedding_model` | `INGESTION__EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model |
| `ingestion.chunk_sentences` | `INGESTION__CHUNK_SENTENCES` | `5` | Sentences per chunk |
| `ingestion.ollama_url` | `INGESTION__OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |

---

## Make targets

```
make help          Show all available targets

make install       Install Python dependencies
make lint          Run ruff linter
make format        Format code with ruff
make test          Run Python test suite

make serve         Start API server (http://localhost:8000)
make dev           Start API + Vite dev server concurrently
make build-web     Build the React SPA into src/frontend/web/dist/

make ingest FILE=  Ingest a document via CLI

make up            Start PostgreSQL + Ollama (Docker)
make pull-model    Pull nomic-embed-text into Ollama (auto-skips if present)
make down          Stop all containers
make up-pgadmin    Start PostgreSQL + Ollama + pgAdmin (http://localhost:5050)
make down-pgadmin  Stop all containers including pgAdmin
```

---

## Document visibility

When uploading via the web UI, the **Personal / Shared** toggle controls who can retrieve the document:

- **Personal** — only visible to the user who uploaded it
- **Shared** — visible to all authenticated users

This is enforced at the database level via PostgreSQL Row-Level Security policies.

---

## Running tests

```bash
# Python
make test

# TypeScript (API client)
cd src/frontend/web && npm test
```
