from pathlib import Path

from pydantic import SecretStr

from src.core.config import ApiSettings, Settings


def test_api_settings_defaults() -> None:
    s = ApiSettings()
    assert s.algorithm == "HS256"
    assert s.token_expire_minutes == 60
    assert s.host == "0.0.0.0"
    assert s.port == 8000


def test_settings_has_api_section() -> None:
    s = Settings()
    assert isinstance(s.api, ApiSettings)


def test_settings_from_yaml_loads_api(tmp_path: Path) -> None:
    yaml_file = tmp_path / "base.yaml"
    yaml_file.write_text("api:\n  port: 9000\n  token_expire_minutes: 30\n")
    s = Settings.from_yaml(yaml_file)
    assert s.api.port == 9000
    assert s.api.token_expire_minutes == 30
