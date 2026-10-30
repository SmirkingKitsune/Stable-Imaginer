import pytest

from imaginer.backends.sizing import fit_area, resolve_size, snap


def _size(**kw: object) -> tuple[int, int]:
    base: dict[str, object] = {
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "default_width": 1024,
        "default_height": 1024,
        "multiple": 32,
        "min_side": 256,
        "max_side": 4096,
    }
    base.update(kw)
    return resolve_size(**base)  # type: ignore[arg-type]


def test_defaults() -> None:
    assert _size() == (1024, 1024)


def test_explicit_sizes_snap_to_multiples() -> None:
    assert _size(width=1000, height=700) == (992, 704)
    assert _size(width=1200) == (1216, 1024)  # missing side falls back to the default


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [("1:1", (1024, 1024)), ("16:9", (1376, 768)), ("9:16", (768, 1376)), ("3:2", (1248, 832))],
)
def test_aspect_ratio_keeps_area(ratio: str, expected: tuple[int, int]) -> None:
    w, h = _size(aspect_ratio=ratio)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0
    assert abs(w * h - 1024 * 1024) / (1024 * 1024) < 0.05


def test_clamping() -> None:
    assert _size(width=64, height=99999) == (256, 4096)
    assert snap(100, 32, 256, 300) == 256
    assert snap(5000, 32, 256, 4010) == 4000  # largest multiple within the limit


def test_aspect_ratio_survives_limits() -> None:
    small = _size(aspect_ratio="16:9", default_width=256, default_height=256)
    assert small == (448, 256)  # scaled up as a whole, not squashed to 16:12
    huge = _size(aspect_ratio="9:16", default_width=4096, default_height=4096, max_side=4096)
    assert huge[1] == 4096 and abs(huge[0] / huge[1] - 9 / 16) < 0.02


def test_edit_follows_its_first_image() -> None:
    # A 4000x2000 photo at a 1 MP default: same 2:1 shape, about 1 MP.
    w, h = _size(base_size=(4000, 2000))
    assert (w, h) == (1440, 736)
    # A small 2:1 input at a tiny default area is scaled up as a whole, not squashed.
    assert _size(base_size=(400, 200), default_width=256, default_height=256) == (512, 256)
    # Explicit sizes and aspect ratios still win over the input's shape.
    assert _size(base_size=(4000, 2000), aspect_ratio="1:1") == (1024, 1024)
    assert _size(base_size=(4000, 2000), width=512, height=512) == (512, 512)


def test_fit_area() -> None:
    assert fit_area(100, 50, 10_000) == (100, 50)
    w, h = fit_area(4000, 3000, 1_200_000)
    assert round(w) == 1265 and round(h) == 949
