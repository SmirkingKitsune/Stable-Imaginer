"""The model library (the web UI's #/models): a card per image model the user imports for
sd-server, with its component files, family, launch arguments, defaults and presets.
Using a card on an engine restarts it, so these routes are for the user only, like the
Engines screen's.
"""

from fastapi import APIRouter, Query, Response, status
from fastapi.responses import FileResponse

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.errors import NotFoundError

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("")
async def list_models(container: ContainerDep) -> dto.ModelLibraryView:
    """Every model card, with whether it is ready to run and which engines run it; the
    families a card can have; every component file with the cards that use it."""
    return await container.models.overview()


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_model(body: dto.ModelCreate, container: ContainerDep) -> dto.ModelCardDetail:
    """A new card: from a model file here (its family detected unless given), from a Hub
    reference (download its components into it next), or empty for a family."""
    return await container.models.create(body)


@router.get("/families")
async def list_families(
    container: ContainerDep,
    hidden: bool = Query(default=False, description="Edit, control and decoder recipes too."),
) -> list[dto.ModelFamily]:
    """The image model families sd-server runs: each recipe's component slots, defaults
    and features, with where on the Hub to get each component."""
    return container.models.families(hidden=hidden)


@router.get("/families/{family_id}")
async def get_family(family_id: str, container: ContainerDep) -> dto.ModelFamily:
    return container.models.family(family_id)


@router.post("/detect")
async def detect_model(body: dto.ModelDetect, container: ContainerDep) -> dto.ModelDetected:
    """Which families a model file (or a Hub reference) looks like, best first, and why."""
    return await container.models.detect(body)


@router.post("/locate")
async def locate_model(body: dto.ModelLocate, container: ContainerDep) -> dto.ModelLocated:
    """Where a file dropped in the browser is on this machine. A browser gives its name and
    size, not its folder, so the model folders are searched for it: downloads, the Hugging
    Face cache, the folders of every card's files, and Downloads."""
    return await container.models.locate(body)


@router.get("/{card_id}")
async def get_model(card_id: str, container: ContainerDep) -> dto.ModelCardDetail:
    """One card, with each component checked on disk, the cards it shares files with, the
    families its file looks like and the images made with it."""
    return await container.models.get(card_id)


@router.patch("/{card_id}")
async def update_model(
    card_id: str, body: dto.ModelUpdate, container: ContainerDep
) -> dto.ModelCardDetail:
    """Change its name, family, component files, launch arguments, defaults, presets,
    prompt note, notes, tags or pin. An engine running it takes new defaults and presets
    at once, and new files when it is restarted."""
    return await container.models.update(card_id, body)


@router.delete("/{card_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(card_id: str, container: ContainerDep) -> Response:
    """Forget the card. Its files stay where they are. Not while an engine runs it."""
    await container.models.delete(card_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{card_id}/use")
async def use_model(
    card_id: str, body: dto.ModelUse, container: ContainerDep
) -> dto.ModelCardDetail:
    """Run it on an sd-server engine. One launched here restarts with its files (409 while
    it is making an image); on one started elsewhere, the card only says what it runs."""
    return await container.models.use(card_id, body)


@router.put("/{card_id}/cover")
async def set_model_cover(
    card_id: str, body: dto.ModelCoverSet, container: ContainerDep
) -> dto.ModelCardDetail:
    """Use an image from a conversation as its cover, or its Hub card's first image."""
    return await container.models.set_cover(card_id, body)


@router.delete("/{card_id}/cover")
async def clear_model_cover(card_id: str, container: ContainerDep) -> dto.ModelCardDetail:
    return await container.models.clear_cover(card_id)


@router.get("/{card_id}/cover", response_class=FileResponse)
async def model_cover(card_id: str, container: ContainerDep) -> FileResponse:
    """The cover image (WebP, at most 768 px on its longer side). Its URL carries a version,
    so it can be cached for good."""
    path = container.models.cover_file(card_id)
    if path is None:
        raise NotFoundError(f"model card {card_id} has no cover")
    return FileResponse(
        path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000"}
    )
