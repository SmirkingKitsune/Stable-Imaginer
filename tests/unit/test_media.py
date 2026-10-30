import pytest

from imaginer.domain.errors import InvalidRequestError, UnsupportedMediaTypeError
from imaginer.services.media import fit_image, inspect_image_bytes
from tests.helpers import image_bytes


def test_png_rgb() -> None:
    info = inspect_image_bytes(image_bytes(12, 7))
    assert (info.mime, info.ext, info.width, info.height) == ("image/png", "png", 12, 7)
    assert info.has_alpha is False


def test_rgba_with_transparency() -> None:
    data = image_bytes(mode="RGBA", color=(10, 20, 30, 255), transparent_pixel=True)
    assert inspect_image_bytes(data).has_alpha is True


def test_rgba_fully_opaque_counts_as_no_alpha() -> None:
    data = image_bytes(mode="RGBA", color=(10, 20, 30, 255))
    assert inspect_image_bytes(data).has_alpha is False


def test_jpeg_and_webp() -> None:
    assert inspect_image_bytes(image_bytes(fmt="JPEG")).mime == "image/jpeg"
    webp = inspect_image_bytes(image_bytes(fmt="WEBP", mode="RGBA", color=(1, 2, 3, 0)))
    assert (webp.mime, webp.ext, webp.has_alpha) == ("image/webp", "webp", True)


def test_mime_comes_from_bytes_not_name() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        inspect_image_bytes(b"this is not an image, whatever the filename says")


def test_unsupported_image_format() -> None:
    with pytest.raises(UnsupportedMediaTypeError, match="unsupported image format"):
        inspect_image_bytes(image_bytes(fmt="TIFF"))


def test_truncated_png() -> None:
    data = image_bytes(64, 64)
    with pytest.raises((InvalidRequestError, UnsupportedMediaTypeError)):
        inspect_image_bytes(data[: len(data) // 2])


def test_fit_image_downscales_and_keeps_alpha() -> None:
    data = image_bytes(400, 200, mode="RGBA", color=(1, 2, 3, 255), transparent_pixel=True)
    fitted = fit_image(data, "image/png", 20_000)
    assert (fitted.width, fitted.height) == (200, 100)
    assert fitted.mime == "image/png"
    assert inspect_image_bytes(fitted.data).has_alpha is True


def test_fit_image_jpeg_when_opaque_and_untouched_when_small() -> None:
    big = image_bytes(400, 200, noise=True)
    fitted = fit_image(big, "image/png", 20_000)
    assert (fitted.mime, fitted.width) == ("image/jpeg", 200)
    small = image_bytes(10, 10)
    assert fit_image(small, "image/png", 20_000).data is small
