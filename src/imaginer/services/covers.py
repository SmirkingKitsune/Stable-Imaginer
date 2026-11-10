"""Cover images for library cards (LoRAs, image models): small WebP thumbnails on disk."""

import io
import os
from pathlib import Path

from PIL import Image

COVER_SIDE = 768  # px, the longer side of a stored cover
COVER_MAX_BYTES = 20 * 1024 * 1024  # a Hub card image larger than this isn't fetched


def thumbnail(data: bytes, side: int = COVER_SIDE) -> tuple[bytes, int, int]:
    """An image as a WebP no larger than ``side`` on its longer side, with its size.
    Blocking; raises PIL's errors (or ValueError) for data that isn't an image."""
    with Image.open(io.BytesIO(data)) as img:
        img.load()
        frame = img.convert("RGBA" if "A" in img.getbands() else "RGB")
    frame.thumbnail((side, side), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    frame.save(out, "WEBP", quality=86, method=4)
    return out.getvalue(), frame.width, frame.height


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
