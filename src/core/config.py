from pathlib import Path

import yaml
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseModel):
    """Ingestion pipeline configuration."""

    embedding_model: str = "nomic-embed-text"
    chunk_sentences: int = 5
    ollama_url: str = "http://localhost:11434"


class DatabaseSettings(BaseModel):
    """PostgreSQL connection configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "rag_db"
    user: str = "rag"
    password: SecretStr = SecretStr("")

    @property
    def dsn(self) -> str:
        """Build asyncpg DSN string."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.name}"
        )


class Settings(BaseSettings):
    """Application settings loaded from YAML then overridden by env vars."""

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    ingestion: IngestionSettings = IngestionSettings()
    database: DatabaseSettings = DatabaseSettings()

    @classmethod
    def from_yaml(cls, path: Path = Path("config/base.yaml")) -> "Settings":
        """Load settings from YAML file, then apply env var overrides."""
        # Load YAML base config
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        return cls(**data)
