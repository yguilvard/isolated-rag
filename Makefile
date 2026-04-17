-include .env
# Fall back to POSTGRES_PASSWORD if DATABASE__PASSWORD not set separately
DATABASE__PASSWORD ?= $(POSTGRES_PASSWORD)
export API__SECRET_KEY
export DATABASE__PASSWORD
export ADMIN_USERNAME
export ADMIN_PASSWORD

COMPOSE = docker compose -f deployment/docker/compose.yaml

.PHONY: help install lint format test ingest serve build-web dev up up-pgadmin down down-pgadmin pull-model

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync
	cd src/frontend/web && npm install

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

test: ## Run tests
	uv run pytest

ingest: ## Ingest a document: make ingest FILE=path/to/doc.pdf
	uv run python -m src.frontend.cli $(FILE)

serve: up pull-model ## Start API server (http://localhost:8000)
	uv run python -m src.frontend.api

build-web: ## Build the React SPA into src/frontend/web/dist/
	cd src/frontend/web && npm run build

dev: up pull-model ## Start API server + Vite dev server concurrently
	uv run python -m src.frontend.api & cd src/frontend/web && npm run dev

up: ## Start postgres + Ollama
	$(COMPOSE) up -d

pull-model: ## Pull the embedding model into Ollama (skips if already present)
	@docker exec $$($(COMPOSE) ps -q ollama) ollama list 2>/dev/null | grep -q nomic-embed-text \
		|| (echo "Pulling nomic-embed-text..." && docker exec $$($(COMPOSE) ps -q ollama) ollama pull nomic-embed-text)

up-pgadmin: ## Start postgres + pgAdmin (http://localhost:5050)
	$(COMPOSE) --profile pgadmin up -d
	@echo "pgAdmin: http://localhost:$${PGADMIN_PORT:-5050}"

down: ## Stop postgres
	$(COMPOSE) down

down-pgadmin: ## Stop postgres + pgAdmin
	$(COMPOSE) --profile pgadmin down
