import uvicorn

from src.core.config import Settings
from src.frontend.api.app import create_app

if __name__ == "__main__":
    settings = Settings.from_yaml()
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        log_level="info",
    )
