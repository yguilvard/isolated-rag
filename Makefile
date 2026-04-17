COMPOSE = docker compose -f deployment/docker/compose.yaml

.PHONY: help install lint format test ingest serve up up-pgadmin down down-pgadmin

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .

test: ## Run tests
	uv run pytest

ingest: ## Ingest a document: make ingest FILE=path/to/doc.pdf
	uv run python -m src.frontend.cli $(FILE)

serve: ## Start API server (http://localhost:8000)
	uv run python -m src.frontend.api

up: ## Start postgres
	$(COMPOSE) up -d

up-pgadmin: ## Start postgres + pgAdmin (http://localhost:5050)
	$(COMPOSE) --profile pgadmin up -d
	@echo "pgAdmin: http://localhost:$${PGADMIN_PORT:-5050}"

down: ## Stop postgres
	$(COMPOSE) down

down-pgadmin: ## Stop postgres + pgAdmin
	$(COMPOSE) --profile pgadmin down
