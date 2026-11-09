"""Request handling shared by the image backends, so engines differ only in transport."""

import asyncio
import hashlib
import json
import os
import struct
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from imaginer.backends.base import GenRequest, ImageInput
from imaginer.backends.sizing import resolve_size
from imaginer.core.config import ImageDefaults, ImagePreset, LoraRef
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
LORAS_SHOWN = 12  # in an error about a LoRA the engine doesn't have
LORA_SCAN_LIMIT = 5000  # folder entries looked at before a LoRA folder's listing stops


def sigmas(value: list[float] | str | None) -> list[float] | None:
    if value is None or isinstance(value, list):
        return value
    try:
        return SIGMA_PRESETS[value]
    except KeyError:
        known = ", ".join(sorted(SIGMA_PRESETS))
        raise ValueError(f"unknown sigma preset '{value}' (known: {known})") from None


def check_sigmas(defaults: ImageDefaults, presets: Mapping[str, ImagePreset]) -> None:
    """Fail fast on an unknown sigma preset name, in the defaults or any preset."""
    sigmas(defaults.sigmas)
    for preset in presets.values():
        sigmas(preset.sigmas)


def find_preset(
    backend: str, presets: Mapping[str, ImagePreset], name: str | None
) -> ImagePreset | None:
    if name is None:
        return None
    try:
        return presets[name]
    except KeyError:
        known = ", ".join(sorted(presets)) or "none"
        raise InvalidRequestError(
            f"backend '{backend}' has no image preset '{name}' (it has: {known})",
            backend=backend,
            preset=name,
        ) from None


def preset_loras(defaults: ImageDefaults, preset: ImagePreset | None) -> list[LoraRef]:
    """The LoRAs a request applies: the preset's when it names any (even none), else the
    defaults'."""
    if preset is not None and preset.loras is not None:
        return list(preset.loras)
    return list(defaults.loras)


def resolve_params(
    req: GenRequest,
    defaults: ImageDefaults,
    *,
    size_multiple: int,
    max_side: int,
    base_size: tuple[int, int] | None = None,
    preset: ImagePreset | None = None,
) -> dict[str, Any]:
    """Our-vocabulary settings for one request: config defaults, overlaid with ``preset``
    (the one ``req.preset`` names, looked up by the adapter), then with the request's own
    overrides. ``loras`` lists the preset's (else the defaults') LoRAs, then the request's;
    the engine adapter merges one named twice, the request's weight winning.

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
    if preset is not None:
        params.update({k: getattr(preset, k) for k in PARAM_KEYS if getattr(preset, k) is not None})
        if preset.steps is not None and preset.sigmas is None:
            params["sigmas"] = None  # a default sigma schedule would decide the steps instead
        params["preset"] = req.preset
    params.update({k: v for k, v in req.params.items() if k in PARAM_KEYS})
    params["negative_prompt"] = (
        req.negative_prompt if req.negative_prompt is not None else defaults.negative_prompt
    )
    fmt = defaults.output_format
    params["output_format"] = "png" if req.alpha and fmt == "jpeg" else fmt
    loras = [*preset_loras(defaults, preset), *req.loras]  # the engine merges repeats
    if loras:
        params["loras"] = [{"path": lora.path, "weight": lora.weight} for lora in loras]
    return {k: v for k, v in params.items() if v is not None}


_HASHES: dict[tuple[str, int, int], str] = {}  # (path, size, mtime) -> SHA-256


def file_sha256(path: Path) -> str:
    """A file's SHA-256, remembered while its size and modification time stay the same, so
    a LoRA is read once per server run rather than once per image. Blocking: run it in a
    worker thread."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    found = _HASHES.get(key)
    if found is None:
        with path.open("rb") as fh:
            found = _HASHES[key] = hashlib.file_digest(fh, "sha256").hexdigest()
    return found


# --- LoRAs ------------------------------------------------------------------------
#
# A LoRA is named by its path as the engine lists it (relative to its LoRA folder, with
# "/"), or by its file name or its name without the extension when only one has it. A
# catalog maps each listed path to the engine's entry for it (``name``: its display name).

Catalog = Mapping[str, Mapping[str, Any]]


def lora_matches(wanted: str, catalog: Catalog) -> list[str]:
    """The listed paths ``wanted`` names: its exact path, else the same path with either
    slash, else a file name, else a name without its extension (one match is an answer;
    several are ambiguous)."""
    if wanted in catalog:
        return [wanted]
    target = wanted.replace("\\", "/")

    def posix(path: str) -> PurePosixPath:
        return PurePosixPath(path.replace("\\", "/"))

    tests: tuple[Callable[[str], bool], ...] = (
        lambda p: str(posix(p)) == target,
        lambda p: posix(p).name == target,
        lambda p: str(catalog[p].get("name") or posix(p).stem) == target,
    )
    for test in tests:
        found = sorted(p for p in catalog if test(p))
        if found:
            return found
    return []


def match_loras(
    backend: str, wanted: list[dict[str, Any]], catalog: Catalog | None, *, unlisted: str
) -> list[dict[str, Any]]:
    """``wanted`` with each path as the engine lists it. A LoRA it doesn't list, or one
    that could be any of several, fails the request; ``unlisted`` says why an engine
    listing none might list none."""
    catalog = catalog or {}
    matched = []
    for lora in wanted:
        found = lora_matches(lora["path"], catalog)
        if len(found) == 1:
            matched.append({**lora, "path": found[0]})
            continue
        if found:
            problem = f"LoRA '{lora['path']}' could be any of {', '.join(found)} on {backend}"
        elif not catalog:
            problem = f"{backend} has no LoRA '{lora['path']}': it lists none ({unlisted})"
        else:
            listed = sorted(catalog)
            shown = ", ".join(listed[:LORAS_SHOWN]) + (", ..." if len(listed) > LORAS_SHOWN else "")
            problem = f"{backend} has no LoRA '{lora['path']}' (it has: {shown})"
        raise InvalidRequestError(problem, backend=backend, lora=lora["path"])
    return matched


def merge_loras(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A LoRA named twice (a preset's, then the request's) keeps its first place and takes
    its last weight; one at weight 0 is left out."""
    merged: dict[str, dict[str, Any]] = {}
    for lora in matched:
        merged[lora["path"]] = {**merged.get(lora["path"], {}), **lora}
    return [lora for lora in merged.values() if lora["weight"] != 0]


def expand_path(path: Path | str) -> Path:
    return Path(os.path.expandvars(path)).expanduser()


def lora_file(folder: Path, listed_path: str) -> Path | None:
    """The file a listed LoRA path names in ``folder``, if it is a file inside it."""
    root = expand_path(folder)
    try:
        path = (root / listed_path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return None
    except (OSError, ValueError):
        return None
    return path


def lora_sha256(folder: Path, listed_path: str) -> str | None:
    """The SHA-256 of a listed LoRA, if its file is in the folder and readable. Blocking."""
    path = lora_file(folder, listed_path)
    try:
        return file_sha256(path) if path is not None else None
    except OSError:
        return None


# Metadata keys trainers write a LoRA's trigger phrase under, most specific first.
TRIGGER_KEYS = ("modelspec.trigger_phrase", "trigger_phrase", "trigger_word", "trigger_words")
TRIGGER_MAX = 300
_HEADER_MAX = 64 * 1024 * 1024
_TRIGGERS: dict[tuple[str, int, int], str | None] = {}


def lora_trigger(path: Path) -> str | None:
    """The trigger phrase a LoRA's safetensors metadata names, if any, remembered while the
    file stays the same. Blocking."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _TRIGGERS:
        _TRIGGERS[key] = _read_trigger(path) if path.suffix.lower() == ".safetensors" else None
    return _TRIGGERS[key]


def _read_trigger(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            if size > _HEADER_MAX:
                return None
            header = json.loads(fh.read(size))
    except (OSError, ValueError, struct.error):
        return None
    meta = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(meta, dict):
        return None
    for key in TRIGGER_KEYS:
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        text = value.strip()
        if text.startswith("["):  # some trainers store a JSON list of words
            try:
                words = json.loads(text)
            except ValueError:
                words = None
            if isinstance(words, list):
                text = ", ".join(str(w).strip() for w in words if str(w).strip())
        return text[:TRIGGER_MAX] or None
    return None


# The trigger words the LoRA library keeps for a LoRA file (the user's, or the ones it
# seeded from the file or its Hub card); None when it has none to say, and the file's
# metadata decides. Called from a worker thread.
WordsSource = Callable[[Path], Sequence[str] | None]


def _words_of(folder: Path, listed_path: str, words: WordsSource | None) -> list[str]:
    path = lora_file(folder, listed_path)
    if path is None:
        return []
    try:
        if words is not None:
            chosen = words(path)
            if chosen is not None:
                return [w for w in chosen if w.strip()]
        trigger = lora_trigger(path)
    except OSError:
        return []
    return [trigger] if trigger else []


def add_phrase(prompt: str, phrase: str) -> str:
    """``prompt`` with ``phrase`` after it, as one more clause."""
    text = prompt.rstrip()
    if not text:
        return phrase
    return (
        f"{text} {phrase}" if text.endswith((".", ",", ";", ":", "!", "?")) else f"{text}, {phrase}"
    )


async def with_triggers(
    folder: Path | None,
    prompt: str,
    loras: list[dict[str, Any]],
    words: WordsSource | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """``prompt`` with the trigger words of each LoRA in ``folder`` that it doesn't have
    yet, and ``loras`` with the ones added as each LoRA's ``trigger``. The words are the
    library's (``words``), else the phrase the file's metadata names."""
    if not loras or folder is None:
        return prompt, loras
    out = []
    for lora in loras:
        added = []
        for word in await asyncio.to_thread(_words_of, folder, lora["path"], words):
            if word.casefold() not in prompt.casefold():
                prompt = add_phrase(prompt, word)
                added.append(word)
        out.append({**lora, "trigger": ", ".join(added)} if added else lora)
    return prompt, out


async def with_sha256(folder: Path | None, loras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``loras`` with the SHA-256 of each one whose file is in ``folder``, for the image's
    record. Hashed in a worker thread; a large LoRA's first hash takes a moment."""
    if not loras or folder is None:
        return loras
    out = []
    for lora in loras:
        digest = await asyncio.to_thread(lora_sha256, folder, lora["path"])
        out.append({**lora, "sha256": digest} if digest else lora)
    return out


def scan_loras(folder: Path, extensions: frozenset[str]) -> dict[str, dict[str, Any]] | None:
    """The files under ``folder`` (subfolders too) with one of ``extensions``, as a catalog
    of paths relative to it; None when it isn't a folder. After ``LORA_SCAN_LIMIT`` entries
    the rest is left out. Blocking."""
    root = expand_path(folder)
    if not root.is_dir():
        return None
    found: dict[str, dict[str, Any]] = {}
    seen = 0
    for here, dirs, names in os.walk(root, onerror=lambda _: None):
        dirs.sort()
        seen += len(dirs) + len(names)
        if seen > LORA_SCAN_LIMIT:
            break
        for name in sorted(names):
            if PurePosixPath(name).suffix.lower() in extensions:
                path = (Path(here) / name).relative_to(root).as_posix()
                found[path] = {"path": path, "name": PurePosixPath(name).stem}
    return found


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
