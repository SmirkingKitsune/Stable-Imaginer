"""Pieces shared by the image tools: common arguments, job running, result shaping."""

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from imaginer.core.config import LoraRef
from imaginer.domain import mentions
from imaginer.domain.errors import ImaginerError, InvalidRequestError
from imaginer.domain.models import SEED_MAX, Artifact, AspectRatio, ImageRequestOptions
from imaginer.orchestration.jobs import JobManager, JobSpec
from imaginer.tools.base import ToolContext, ToolOutcome

if TYPE_CHECKING:
    from imaginer.backends.registry import BackendRegistry


class ImageOptions(BaseModel):
    """Output options both image tools accept."""

    model_config = ConfigDict(extra="forbid")

    negative_prompt: str | None = Field(
        default=None, max_length=2000, description="Things that must not appear."
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="Shape of the image. Omit for the default."
    )
    width: int | None = Field(
        default=None, ge=256, le=4096, description="Exact width in pixels; rarely needed."
    )
    height: int | None = Field(
        default=None, ge=256, le=4096, description="Exact height in pixels; rarely needed."
    )
    n: int = Field(default=1, ge=1, le=4, description="How many variations to make.")
    seed: int | None = Field(
        default=None, ge=0, le=SEED_MAX, description="Reuse a seed to reproduce an image."
    )
    transparent: bool = Field(
        default=False,
        description="Transparent background (RGBA), e.g. for stickers, logos or cut-outs.",
    )


def apply_user_options[T: ImageOptions](args: T, options: ImageRequestOptions | None) -> T:
    """The user's picks for this message win over what the model asked for."""
    if options is None:
        return args
    update: dict[str, Any] = {}
    if options.width is not None and options.height is not None:
        update |= {"width": options.width, "height": options.height, "aspect_ratio": None}
    elif options.aspect_ratio is not None:
        update |= {"aspect_ratio": options.aspect_ratio, "width": None, "height": None}
    if options.n is not None:
        update["n"] = options.n
    if options.seed is not None:
        update["seed"] = options.seed
    if options.negative_prompt is not None:
        update["negative_prompt"] = options.negative_prompt or None
    if options.transparent is not None:
        update["transparent"] = options.transparent
    return args.model_copy(update=update)


def user_params(options: ImageRequestOptions | None) -> dict[str, Any]:
    """Sampler overrides only the user sets (the tools don't offer them to the model)."""
    if options is None:
        return {}
    params = {"steps": options.steps, "cfg_scale": options.cfg_scale}
    return {k: v for k, v in params.items() if v is not None}


def user_preset(
    options: ImageRequestOptions | None,
    registry: "BackendRegistry | None" = None,
    task: Literal["t2i", "edit"] = "t2i",
    *,
    mask: bool = False,
) -> dict[str, Any]:
    """The backend preset the user picked, as keyword arguments for the lease and the job
    (only the user picks one: its LoRAs and settings aren't offered to the model). A preset
    of an engine's edit card is for its edits, and one of its model card for the rest (and
    for edits with a ``mask``, when only the model card takes one): a request the other card
    runs goes without it."""
    if options is None or not options.preset:
        return {}
    if registry is not None:
        mine = registry.image_preset(task, options.backend, options.preset, mask=mask)
        tasks: tuple[Literal["t2i", "edit"], ...] = ("t2i", "edit")
        if mine is None and any(
            registry.image_preset(card, options.backend, options.preset) for card in tasks
        ):
            return {}
    return {"preset": options.preset}


def user_loras(options: ImageRequestOptions | None) -> tuple[LoraRef, ...]:
    """The LoRAs the user added to this message (only the user picks them, like presets)."""
    if options is None or not options.loras:
        return ()
    return tuple(LoraRef(path=x.path, weight=x.weight) for x in options.loras)


def image_prompt(prompt: str) -> str:
    """The prompt the engine gets: without LoRA mentions, which only the user's own
    message applies (a raw prompt has none left by now; the LLM's are dropped), and without
    the note that lists them, should the LLM have copied it."""
    text = mentions.strip(mentions.without_note(prompt))
    if not text:
        raise InvalidRequestError(
            "the prompt is only LoRA mentions: describe the image (the user picks LoRAs)"
        )
    return text


def loras_requested(loras: tuple[LoraRef, ...]) -> dict[str, Any]:
    """The user's LoRAs as the job records what was asked for."""
    return {"loras": [x.model_dump() for x in loras]} if loras else {}


def sampler_params(ctx: ToolContext) -> dict[str, Any]:
    """The call's sampler overrides: the user's, plus any a quality check picked."""
    return {**ctx.sampler, **user_params(ctx.image_options)}


def user_options_note(options: ImageRequestOptions | None) -> dict[str, Any]:
    """Tells the model which settings came from the user rather than from its call."""
    picked = options.model_dump(exclude_none=True) if options is not None else {}
    return {"user_options": picked} if picked else {}


def outputs_summary(artifacts: list[Artifact]) -> list[dict[str, Any]]:
    return [
        {
            "label": a.label,
            "width": a.width,
            "height": a.height,
            "transparent": bool(a.has_alpha),
            "seed": (a.metadata or {}).get("seed"),
        }
        for a in artifacts
    ]


async def run_image_job(
    jobs: JobManager, spec: JobSpec, ctx: ToolContext, **extra: Any
) -> ToolOutcome:
    """Submit, wait, and describe the result for the model.

    On failure or cancellation, images the job already finished (``n`` > 1) are kept,
    attached, and listed in the error so the model can mention them.
    """
    job_id = await jobs.submit(spec, ctx.emit)
    try:
        artifacts = await jobs.wait(job_id)
    except ImaginerError as exc:
        outcome = ToolOutcome.error(exc.code, exc.detail, job_id=job_id)
        partial = await jobs.outputs(job_id)
        if partial:
            outcome.content["error"]["images"] = outputs_summary(partial)
            outcome.artifact_ids = [a.id for a in partial]
        return outcome
    return ToolOutcome(
        content={
            "status": "succeeded",
            "job_id": job_id,
            **extra,
            "images": outputs_summary(artifacts),
            "note": "The user can see these images now.",
        },
        artifact_ids=[a.id for a in artifacts],
    )
