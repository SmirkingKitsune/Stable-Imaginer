"""Output size resolution shared by image backends."""

import math

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
}


def snap(value: float, multiple: int, low: int, high: int) -> int:
    """Round to the nearest multiple, then clamp into [low, high] on multiples."""
    snapped = max(multiple, round(value / multiple) * multiple)
    low_m = math.ceil(low / multiple) * multiple
    high_m = max(low_m, high // multiple * multiple)
    return min(max(snapped, low_m), high_m)


def resolve_size(
    *,
    width: int | None,
    height: int | None,
    aspect_ratio: str | None,
    default_width: int,
    default_height: int,
    multiple: int,
    min_side: int,
    max_side: int,
    base_size: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Explicit sizes win. Otherwise the shape comes from ``aspect_ratio``, then from
    ``base_size`` (an edit's first input image), then the default, always at no more than
    the default's pixel area. Results are multiples of ``multiple`` (Qwen-Image-2.1 needs
    32) within [min_side, max_side]."""
    w: float
    h: float
    area = default_width * default_height
    if width is not None or height is not None:
        w = width if width is not None else default_width
        h = height if height is not None else default_height
    elif aspect_ratio is not None:
        rw, rh = ASPECT_RATIOS[aspect_ratio]
        w = math.sqrt(area * rw / rh)
        h = area / w
        w, h = _within(w, h, min_side, max_side)
    elif base_size is not None:
        w, h = fit_area(*base_size, area)
        w, h = _within(w, h, min_side, max_side)
    else:
        w, h = default_width, default_height
    return (
        snap(w, multiple, min_side, max_side),
        snap(h, multiple, min_side, max_side),
    )


def _within(w: float, h: float, min_side: int, max_side: int) -> tuple[float, float]:
    """Scale both sides together so the limits never distort the shape."""
    if min(w, h) < min_side:
        factor = min_side / min(w, h)
        w, h = w * factor, h * factor
    if max(w, h) > max_side:
        factor = max_side / max(w, h)
        w, h = w * factor, h * factor
    return w, h


def fit_area(width: float, height: float, max_area: float) -> tuple[float, float]:
    """Scale (width, height) down, keeping its aspect ratio, so it covers at most max_area."""
    area = width * height
    if area <= max_area:
        return width, height
    factor = math.sqrt(max_area / area)
    return width * factor, height * factor
