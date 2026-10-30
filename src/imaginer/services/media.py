"""Image inspection. The MIME type always comes from the bytes, never from the client."""

import base64
import io
from dataclasses import dataclass
from typing import BinaryIO

from PIL import Image, UnidentifiedImageError

from imaginer.domain.errors import InvalidRequestError, UnsupportedMediaTypeError

# Pillow format name -> (MIME type, file extension)
IMAGE_FORMATS: dict[str, tuple[str, str]] = {
    "PNG": ("image/png", "png"),
    "JPEG": ("image/jpeg", "jpg"),
    "WEBP": ("image/webp", "webp"),
    "GIF": ("image/gif", "gif"),
    "BMP": ("image/bmp", "bmp"),
}


EXTENSIONS: dict[str, str] = dict(IMAGE_FORMATS.values())


@dataclass(frozen=True, slots=True)
class ImageInfo:
    mime: str
    ext: str
    width: int
    height: int
    has_alpha: bool  # carries real transparency, not merely an opaque alpha band


def inspect_image(fp: BinaryIO) -> ImageInfo:
    """Identify an image and measure it. Blocking: run it in a worker thread."""
    try:
        with Image.open(fp) as im:
            fmt = im.format or ""
            if fmt not in IMAGE_FORMATS:
                raise UnsupportedMediaTypeError(
                    f"unsupported image format {fmt or 'unknown'}; "
                    f"accepted: {', '.join(sorted(IMAGE_FORMATS))}"
                )
            mime, ext = IMAGE_FORMATS[fmt]
            im.load()  # full decode: rejects truncated/corrupt files before engines see them
            width, height = im.size
            return ImageInfo(
                mime=mime, ext=ext, width=width, height=height, has_alpha=_has_alpha(im)
            )
    except UnidentifiedImageError as exc:
        raise UnsupportedMediaTypeError("not a recognised image file") from exc
    except Image.DecompressionBombError as exc:
        raise InvalidRequestError("image dimensions are too large") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise InvalidRequestError(f"corrupt or truncated image: {exc}") from exc


def inspect_image_bytes(data: bytes) -> ImageInfo:
    return inspect_image(io.BytesIO(data))


def encode_for_llm(data: bytes, max_side: int) -> str:
    """A ``data:`` URL of the image, downscaled so its longer side is at most ``max_side``.

    Transparent images stay PNG; everything else becomes JPEG to save prompt bandwidth.
    Blocking: run it in a worker thread.
    """
    with Image.open(io.BytesIO(data)) as source:
        im = source.convert("RGBA") if _has_alpha(source) else source.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if im.mode == "RGBA":
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        im.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


@dataclass(frozen=True, slots=True)
class FittedImage:
    data: bytes
    mime: str
    width: int
    height: int


def fit_image(data: bytes, mime: str, max_pixels: int) -> FittedImage:
    """Downscale an image to at most ``max_pixels``, keeping aspect ratio and transparency.

    Images that already fit are returned untouched. Blocking: run it in a worker thread.
    """
    with Image.open(io.BytesIO(data)) as source:
        width, height = source.size
        if width * height <= max_pixels:
            return FittedImage(data=data, mime=mime, width=width, height=height)
        scale = (max_pixels / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        alpha = _has_alpha(source)
        im = source.convert("RGBA" if alpha else "RGB").resize(size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if alpha:
        im.save(buf, format="PNG")
        mime = "image/png"
    else:
        im.save(buf, format="JPEG", quality=95)
        mime = "image/jpeg"
    return FittedImage(data=buf.getvalue(), mime=mime, width=size[0], height=size[1])


def _has_alpha(im: Image.Image) -> bool:
    if "A" in im.getbands():
        low = im.getchannel("A").getextrema()[0]
        assert isinstance(low, int | float)
        return low < 255
    return im.mode == "P" and "transparency" in im.info
