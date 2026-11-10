"""Agent portraits on disk: square WebP images, one file per portrait an agent has.

``<data_root>/agent_portraits/<agent id>/<key>.webp``, where the key is ``hub`` (its Hub
uploader's avatar), ``upload`` (an image someone gave it) or ``self-<n>`` (its n-th
self-portrait). Their URLs carry a version, so they can be cached for good.
"""

import io
import re
from pathlib import Path

from PIL import Image

PORTRAIT_SIDE = 512  # px: what the agent card's hero shows at most, twice over
PORTRAIT_MAX_BYTES = 20 * 1024 * 1024  # an avatar or upload larger than this isn't read
KEY = re.compile(r"^(hub|upload|self-(\d{1,6}))$")
_ID = re.compile(r"^[0-9a-z-]{1,64}$")


def square(data: bytes, side: int = PORTRAIT_SIDE) -> bytes:
    """The image cropped to its centre square, at most ``side`` px, as WebP. Blocking;
    raises PIL's errors (or ValueError) for data that isn't an image."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        frame = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    width, height = frame.size
    edge = min(width, height)
    if edge < 1:
        raise ValueError("the image is empty")
    left, top = (width - edge) // 2, (height - edge) // 2
    frame = frame.crop((left, top, left + edge, top + edge))
    if edge > side:
        frame = frame.resize((side, side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    frame.save(out, "WEBP", quality=88, method=4)
    return out.getvalue()


def path(root: Path, agent_id: str, key: str) -> Path | None:
    """Where a portrait is kept, or None for an id or key that can't be one."""
    if not _ID.fullmatch(agent_id) or not KEY.fullmatch(key):
        return None
    return root / agent_id / f"{key}.webp"


def self_version(key: str | None) -> int | None:
    found = KEY.fullmatch(key or "")
    return int(found.group(2)) if found and found.group(2) else None
