from fastapi import APIRouter

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto

router = APIRouter(prefix="/v1/backends", tags=["backends"])


@router.get("")
async def list_backends(container: ContainerDep) -> list[dto.BackendInfo]:
    """Every configured backend with live health and advertised capabilities.

    Capabilities are the configured upper bound narrowed by what the engine reports, so a
    missing capability means it really is unavailable.
    """
    return await container.registry.describe()
