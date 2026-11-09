import io
import json
import os
import struct
from pathlib import Path
from typing import Any

from PIL import Image


def image_bytes(
    width: int = 8,
    height: int = 8,
    *,
    fmt: str = "PNG",
    mode: str = "RGB",
    color: tuple[int, ...] = (200, 30, 30),
    transparent_pixel: bool = False,
    noise: bool = False,
) -> bytes:
    """A tiny in-memory image. ``transparent_pixel`` punches one fully clear pixel (RGBA);
    ``noise`` fills it with random pixels so it does not compress."""
    if noise:
        im = Image.frombytes(mode, (width, height), os.urandom(width * height * len(mode)))
    else:
        im = Image.new(mode, (width, height), color)
    if transparent_pixel:
        im.putpixel((0, 0), (0, 0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def parse_sse(text: str) -> list[tuple[str, int, dict[str, Any]]]:
    """(event, id, data) for every event in a text/event-stream body; pings are skipped."""
    events = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, str] = {}
        data_lines = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if name == "data":
                data_lines.append(value)
            else:
                fields[name] = value
        if "event" in fields:
            events.append((fields["event"], int(fields["id"]), json.loads("\n".join(data_lines))))
    return events


def safetensors_file(path: Path, metadata: dict[str, str]) -> Path:
    """A one-tensor safetensors file with ``metadata`` in its header, as a LoRA trainer
    leaves its trigger phrase."""
    header = {"__metadata__": metadata, "w": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0")
    return path
