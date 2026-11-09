"""The LoRA library (the web UI's #/loras): every LoRA the image engines can apply, with
what the user adds to each. Moving and deleting change files on this machine and the
presets that name them, so these routes are for the user only, like the Engines screen's.
"""

from fastapi import APIRouter, Response, status
from fastapi.responses import FileResponse

from imaginer.api.deps import ContainerDep
from imaginer.domain import models as dto
from imaginer.domain.errors import NotFoundError

router = APIRouter(prefix="/v1/loras", tags=["loras"])


@router.get("")
async def list_loras(container: ContainerDep) -> dto.LoraLibrary:
    """Every LoRA, one per file (by SHA-256) however many folders it is in, with its
    engines, compatibility, usage and what the user added; and the models the engines run."""
    return await container.loras.overview()


@router.get("/{lora_id}")
async def get_lora(lora_id: str, container: ContainerDep) -> dto.LoraDetail:
    """One LoRA, with its compatibility per model, the presets that can hold it, the
    conversations that used it and its files."""
    return await container.loras.get(lora_id)


@router.patch("/{lora_id}")
async def update_lora(
    lora_id: str, body: dto.LoraUpdate, container: ContainerDep
) -> dto.LoraDetail:
    """Change its display name, default weight, trigger words, notes, tags, pin, lineage,
    or the user's call on a model."""
    return await container.loras.update(lora_id, body)


@router.post("/{lora_id}/move")
async def move_lora(lora_id: str, body: dto.LoraMove, container: ContainerDep) -> dto.LoraDetail:
    """Move its file into another folder of its LoRA folder. Presets and defaults that name
    its old path are pointed at the new one."""
    return await container.loras.move(lora_id, body.folder)


@router.delete("/{lora_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lora(lora_id: str, container: ContainerDep) -> Response:
    """Delete its file here, and take it out of the presets and defaults that name it."""
    await container.loras.delete(lora_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{lora_id}/presets")
async def set_lora_preset(
    lora_id: str, body: dto.LoraPresetSet, container: ContainerDep
) -> dto.LoraDetail:
    """Put it in one of an engine's presets at a weight, take it out (weight null), or
    make a new preset with it."""
    return await container.loras.set_preset(lora_id, body)


@router.put("/{lora_id}/cover")
async def set_lora_cover(
    lora_id: str, body: dto.LoraCoverSet, container: ContainerDep
) -> dto.LoraDetail:
    """Use an image from a conversation as its cover, or its Hub card's first image."""
    return await container.loras.set_cover(lora_id, body)


@router.delete("/{lora_id}/cover")
async def clear_lora_cover(lora_id: str, container: ContainerDep) -> dto.LoraDetail:
    return await container.loras.clear_cover(lora_id)


@router.get("/{lora_id}/cover", response_class=FileResponse)
async def lora_cover(lora_id: str, container: ContainerDep) -> FileResponse:
    """The cover image (WebP, at most 768 px on its longer side). Its URL carries a version,
    so it can be cached for good."""
    path = container.loras.cover_file(lora_id)
    if path is None:
        raise NotFoundError(f"LoRA {lora_id} has no cover")
    return FileResponse(
        path, media_type="image/webp", headers={"Cache-Control": "private, max-age=31536000"}
    )
