"""Pieces shared by the image tools: common arguments, job running, result shaping."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from imaginer.domain.errors import ImaginerError
from imaginer.domain.models import SEED_MAX, Artifact, AspectRatio, ImageRequestOptions
from imaginer.orchestration.jobs import JobManager, JobSpec
from imaginer.tools.base import ToolContext, ToolOutcome


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
