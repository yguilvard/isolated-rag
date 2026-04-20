from typing import Annotated

from fastapi import APIRouter, Depends

from src.core.config import Settings
from src.frontend.api.deps import get_current_user, get_settings
from src.frontend.api.schemas import UserClaims

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(
    _user: Annotated[UserClaims, Depends(get_current_user)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> list[dict[str, str]]:
    """Return the list of available generative chat providers.

    Args:
        settings: Application settings containing the chat_providers list.

    Returns:
        List of dicts with ``id`` and ``label`` for each configured provider.
    """
    return [{"id": p.id, "label": p.label} for p in settings.chat_providers]
