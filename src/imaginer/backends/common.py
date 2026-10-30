"""Request handling shared by the image backends, so engines differ only in transport."""

import asyncio
from collections.abc import Sequence
from typing import Any

from imaginer.backends.base import GenRequest, ImageInput
from imaginer.backends.sizing import resolve_size
from imaginer.core.config import ImageDefaults
from imaginer.domain.errors import InvalidRequestError
from imaginer.services.media import FittedImage, fit_image

MIN_SIDE = 256
# Named sigma schedules (docs/qwen_image_2.1.md in stable-diffusion.cpp), expanded locally
# because engine APIs only accept explicit lists. The trailing 0.0 is the final sigma.
SIGMA_PRESETS: dict[str, list[float]] = {
    "qwen-image-2.1-turbo": [
        1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0,
    ],
}  # fmt: skip
PARAM_KEYS = ("steps", "cfg_scale", "sampler", "scheduler", "flow_shift", "sigmas")


def sigmas(value: list[float] | str | None) -> list[float] | None:
    if value is None or isinstance(value, list):
        return value
    try:
        return SIGMA_PRESETS[value]
    except KeyError:
        known = ", ".join(sorted(SIGMA_PRESETS))
        raise ValueError(f"unknown sigma preset '{value}' (known: {known})") from None


def resolve_params(
    req: GenRequest,
    defaults: ImageDefaults,
    *,
    size_multiple: int,
    max_side: int,
    base_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Our-vocabulary settings for one request: config defaults overlaid with overrides.

    ``base_size`` (an edit's first image) sets the output shape when the request names no
    size or aspect ratio. Unset values are dropped, so engines keep their own defaults.
    """
    width, height = resolve_size(
        width=req.width,
        height=req.height,
        aspect_ratio=req.aspect_ratio,
        base_size=base_size,
        default_width=defaults.width,
        default_height=defaults.height,
        multiple=size_multiple,
        min_side=MIN_SIDE,
        max_side=max_side,
    )
    params: dict[str, Any] = {
        "width": width,
        "height": height,
        "steps": defaults.steps,
        "cfg_scale": defaults.cfg_scale,
        "sampler": defaults.sampler,
        "scheduler": defaults.scheduler,
        "flow_shift": defaults.flow_shift,
        "sigmas": defaults.sigmas,
    }
    params.update({k: v for k, v in req.params.items() if k in PARAM_KEYS})
    params["negative_prompt"] = (
        req.negative_prompt if req.negative_prompt is not None else defaults.negative_prompt
    )
    fmt = defaults.output_format
    params["output_format"] = "png" if req.alpha and fmt == "jpeg" else fmt
    return {k: v for k, v in params.items() if v is not None}


def styled_prompt(req: GenRequest, rgba_template: str) -> str:
    """Qwen-Image-2.1 decides transparency from the prompt, using its official phrasing."""
    if req.alpha:
        return rgba_template.format(prompt=req.prompt.strip().rstrip("."))
    return req.prompt


def default_max_ref_pixels(defaults: ImageDefaults, configured: int | None) -> int:
    return configured or max(defaults.width * defaults.height, 1024 * 1024)


async def prepare_inputs(
    backend: str,
    images: Sequence[ImageInput],
    mask: ImageInput | None,
    *,
    max_inputs: int,
    max_pixels: int,
) -> list[FittedImage]:
    """Validate and downscale edit inputs. The mask, if any, goes last.

    Reference images become model tokens, so an unscaled phone photo would be huge.
    """
    inputs = [*images, *([mask] if mask is not None else [])]
    if not images:
        raise InvalidRequestError("edit needs at least one input image")
    if len(inputs) > max_inputs:
        raise InvalidRequestError(f"{backend} accepts at most {max_inputs} reference images")
    return [
        await asyncio.to_thread(fit_image, image.data, image.mime, max_pixels) for image in inputs
    ]


def mask_prompt(prompt: str, instruction: str) -> str:
    return f"{prompt.rstrip()} {instruction}"
