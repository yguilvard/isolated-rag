from pathlib import Path
from typing import Any, ClassVar, Tuple, Type

import yaml
from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class ChatProviderConfig(BaseModel):
    """A single named generative model served by a dedicated Ollama instance."""

    id: str
    label: str
    model: str
    ollama_url: str


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


class ApiSettings(BaseModel):
    """FastAPI server configuration."""

    secret_key: SecretStr = SecretStr("")
    algorithm: str = "HS256"
    token_expire_minutes: int = 60
    host: str = "0.0.0.0"
    port: int = 8000


class _YamlSettingsSource(PydanticBaseSettingsSource):
    """YAML file settings source — lower priority than env vars."""

    def __init__(
        self,
        settings_cls: Type[BaseSettings],
        yaml_path: Path = Path("config/base.yaml"),
    ) -> None:
        super().__init__(settings_cls)
        # Load YAML base config from file
        self._data: dict[str, Any] = {}
        if yaml_path.exists():
            self._data = yaml.safe_load(yaml_path.read_text()) or {}

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Return value, field name, and whether the value is complex."""
        value = self._data.get(field_name)
        return value, field_name, self.field_is_complex(field)

    def __call__(self) -> dict[str, Any]:
        """Return all YAML-sourced values for pydantic-settings to merge."""
        return {k: v for k, v in self._data.items() if v is not None}


class Settings(BaseSettings):
    """Application settings: env vars override YAML, which overrides defaults."""

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    # Holds the active YAML path so settings_customise_sources can access it
    _yaml_path: ClassVar[Path] = Path("config/base.yaml")

    ingestion: IngestionSettings = IngestionSettings()
    database: DatabaseSettings = DatabaseSettings()
    api: ApiSettings = ApiSettings()
    chat_providers: list[ChatProviderConfig] = []

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Priority: init kwargs > env vars > YAML file > field defaults."""
        return (init_settings, env_settings, _YamlSettingsSource(settings_cls, cls._yaml_path))

    @classmethod
    def from_yaml(cls, path: Path = Path("config/base.yaml")) -> "Settings":
        """Load settings: env vars take precedence over YAML defaults."""
        cls._yaml_path = path
        return cls()
