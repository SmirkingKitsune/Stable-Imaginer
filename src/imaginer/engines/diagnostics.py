"""What went wrong with an engine this server started, told so the chat model can explain
it and suggest a fix: how its process ended, the files it ran, the lines of its log that
matter, and hints from the failures we know.

sd-server says little over HTTP when it fails: a crash ends the process, and the next
request only finds nothing listening. Its log says why (a failed check, memory running out,
a file it couldn't read), so the report is made from the log tail the supervisor keeps.
"""

import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from imaginer.backends.base import EngineReport
from imaginer.engines.supervisor import EngineProcess, LogLine
from imaginer.image_models import recipes

TAIL_LINES = 20  # of the log, as the report shows it
ERROR_LINES = 8
WARNING_LINES = 4
LINE_CHARS = 300
FLAGS = {flag: slot for slot, flag, _ in recipes.SLOTS}

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\[K")
_PROGRESS = re.compile(r"^\s*\|.*\|\s*\d+/\d+")  # its loading and sampling bars
_SOURCE = re.compile(r"\s+---\s+\S+:\d+\s*$")  # sd.cpp's " --- model_loader.cpp:226"
_SOURCE_PREFIX = re.compile(r"^\S+\.(?:c|cc|cpp|cu|h|hpp):\d+:\s*")  # "D:\...\ggml.c:3378: "
_ERROR_LEVEL = re.compile(r"^\[E\]")
_WARNING_LEVEL = re.compile(r"^\[W\]")
_VERSION = re.compile(r"\bVersion:\s*(.+?)\s*$")
# Where sd-server got to with an image: it starts one, encodes an edit's reference images
# with the VAE, the prompt with the text encoder, samples with the diffusion model, then
# decodes with the VAE.
_IMAGE_START = re.compile(r"\bgenerate_image \d+x\d+")
_SAMPLING = re.compile(r"\bgenerating image: \d+/\d+")
_DECODING = re.compile(r"\bdecoding \d+ latents|\bdecode_first_stage")
_VAE_ENCODED = re.compile(r"\bencode_first_stage")
VAE_SLOTS = frozenset({"vae", "tae"})
VAE_IDLE = "the VAE hadn't run"

# What a log line usually means, checked in order (the last one only when none matched).
HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"ggml_can_mul_mat|ggml_are_same_shape|ggml_can_repeat|shape mismatch|wrong shape",
            re.IGNORECASE,
        ),
        "A tensor shape mismatch: the model's files don't fit together. Usually one of them "
        "(often the text encoder or the VAE) belongs to another model, or the model card "
        "names the wrong family.",
    ),
    (
        re.compile(
            r"out of memory|cudaMalloc failed|failed to allocate|ErrorOutOfDeviceMemory|"
            r"not enough memory",
            re.IGNORECASE,
        ),
        "Memory ran out (GPU or RAM): a smaller image, fewer reference images, a quantized "
        "model or offloading weights to the CPU may help.",
    ),
    (
        re.compile(r"tensor\b.*\bnot found|missing tensor|unknown tensor", re.IGNORECASE),
        "The weights lack tensors the model needs: a file in the wrong slot, from another "
        "model, or incomplete.",
    ),
    (
        re.compile(
            r"get sd version|failed to load|load tensors from model loader failed|"
            r"invalid magic|unsupported (?:model|format|file|tensor type)|unknown model",
            re.IGNORECASE,
        ),
        "sd-server couldn't read a model file: it may be incomplete (download it again), "
        "in a format this sd.cpp build doesn't read, or a model it doesn't support.",
    ),
    (
        re.compile(
            r"illegal memory access|CUDA error(?!: out of memory)|device-side assert|"
            r"vulkan error",
            re.IGNORECASE,
        ),
        "A GPU error. If it happens again with the same files and settings, the sd.cpp "
        "runtime or the GPU driver may need updating.",
    ),
)
_FALLBACK = (
    re.compile(r"GGML_ASSERT|assert(?:ion)? failed|\babort", re.IGNORECASE),
    "sd-server stopped at a failed internal check: something in the request or the files "
    "is what this build can't handle. The failed check is in the log.",
)
EXITED = (
    "The engine is not running any more. It has to be started again (Engines screen) "
    "before it can make images, once the cause is fixed: otherwise it fails the same way."
)
STOPPED = (
    "The engine was stopped (Engines screen): it has to be started again before it can make images."
)
UNUSUAL = "Check the unusual files first: one of them is the likely cause."

# Exit codes worth naming: abort() on Windows and POSIX, crashes, and the OOM killer.
_EXITS = {
    3: "aborted",
    134: "aborted",
    -6: "aborted",
    0xC0000409: "aborted at a failed check",
    0xC0000005: "access violation",
    0xC00000FD: "stack overflow",
    0xC0000017: "out of memory",
    -11: "segmentation fault",
    139: "segmentation fault",
    -9: "killed, maybe for running out of memory",
    137: "killed, maybe for running out of memory",
}


def exit_text(code: int | None) -> str | None:
    """``exit code 0xC0000409, aborted at a failed check``: Windows status codes in hex."""
    if code is None:
        return None
    shown = f"0x{code:08X}" if code >= 0x80000000 else str(code)
    meaning = _EXITS.get(code)
    return f"exit code {shown}" + (f", {meaning}" if meaning else "")


def clean(text: str) -> str | None:
    """A log line as the report shows it: no terminal codes, progress bars or source
    locations, and not too long. None for a line not worth showing."""
    text = _ANSI.sub("", text).rstrip()
    if not text.strip() or text.startswith("$ ") or _PROGRESS.match(text):
        return None
    text = _SOURCE.sub("", text)
    return text if len(text) <= LINE_CHARS else text[: LINE_CHARS - 1] + "…"


def run_files(argv: Sequence[str]) -> dict[str, str]:
    """The model files a run was started with, by slot."""
    return {FLAGS[flag]: value for flag, value in itertools.pairwise(argv) if flag in FLAGS}


def _relative(path: str, models_dir: Path) -> str | None:
    """``path`` as owner/repo/file when it is in the models folder (where the Hub's
    downloads go), else None."""
    try:
        return Path(path).absolute().relative_to(models_dir.absolute()).as_posix()
    except ValueError:
        return None


def failed_while(texts: Sequence[str]) -> str | None:
    """What sd-server was doing with the image it was making when it failed (None: it
    wasn't making one, e.g. it failed loading)."""
    starts = [i for i, text in enumerate(texts) if _IMAGE_START.search(text)]
    if not starts:
        return None
    image = texts[starts[-1] :]
    if any(_DECODING.search(t) for t in image):
        return "decoding the image (the VAE)"
    if any(_SAMPLING.search(t) for t in image):
        if any(_VAE_ENCODED.search(t) for t in image):
            return "sampling (the diffusion model, on the encoded prompt and reference images)"
        return f"sampling (the diffusion model, on the text encoder's output; {VAE_IDLE})"
    return "encoding the prompt (the text encoder), or an edit's reference images (the VAE)"


def unusual_files(
    files: Mapping[str, str],
    family: recipes.Family | None,
    models_dir: Path,
    *,
    skip: frozenset[str] = frozenset(),
) -> list[str]:
    """The downloaded files that come from none of the repos their family usually takes
    that slot from, leaving out the ``skip`` slots (files the failure didn't involve). A
    hint, not a verdict: a file can be right from elsewhere."""
    if family is None:
        return []
    out = []
    for slot, path in files.items():
        sources = [s.split(":")[0].strip("/") for s in family.sources.get(slot, ())]
        shown = _relative(path, models_dir)
        if not sources or shown is None or slot in skip:
            continue
        if any(shown.casefold().startswith(src.casefold() + "/") for src in sources):
            continue
        what = recipes.SLOT_WHAT.get(slot, slot)
        out.append(
            f"{slot} ({what}) {shown} is from none of the repos {family.name} usually takes "
            f"it from: {', '.join(sources)}"
        )
    return out


def hints(texts: Iterable[str]) -> list[str]:
    lines = list(texts)
    found = [hint for pattern, hint in HINTS if any(pattern.search(t) for t in lines)]
    if not found and any(_FALLBACK[0].search(t) for t in lines):
        found.append(_FALLBACK[1])
    return found


def _collapse(texts: Iterable[str]) -> list[str]:
    """Repeated lines once."""
    out: list[str] = []
    for text in texts:
        if not out or out[-1] != text:
            out.append(text)
    return out


def _last_unique(texts: Iterable[str], most: int) -> list[str]:
    seen: list[str] = []
    for text in reversed(list(texts)):
        if text not in seen:
            seen.append(text)
        if len(seen) == most:
            break
    return seen[::-1]


def report(
    proc: EngineProcess,
    *,
    family: recipes.Family | None,
    models_dir: Path,
    since: datetime,
) -> EngineReport | None:
    """What the engine's last run shows about a failure of work that started at ``since``;
    None when it has never run here. A running engine's log counts from ``since`` (what it
    said earlier is about earlier work); an exited one's to its end, whenever that was."""
    run = proc.run
    if run is None:
        return None
    if proc.state == "stopped":
        return EngineReport(state="stopped", details={"state": "stopped", "hints": [STOPPED]})
    state = "exited" if proc.state == "exited" else "running"
    lines: list[tuple[LogLine, str]] = [
        (line, text)
        for line in run.lines
        if (state == "exited" or line.at >= since) and (text := clean(line.text)) is not None
    ]
    errors = [t for line, t in lines if line.error or _ERROR_LEVEL.match(t)]
    warnings = [t for line, t in lines if _WARNING_LEVEL.match(t) and t not in errors]
    files = run_files(run.argv)
    stage = failed_while([t for _, t in lines])
    # Sampling text-to-image doesn't use the VAE: a VAE from elsewhere can't be the cause.
    idle = VAE_SLOTS if stage is not None and VAE_IDLE in stage else frozenset[str]()
    unusual = unusual_files(files, family, models_dir, skip=idle)
    found = hints(t for _, t in lines)
    if unusual and found:
        found.append(UNUSUAL)
    if state == "exited":
        found.append(EXITED)
    detected = next(
        (m.group(1) for line in run.lines if (m := _VERSION.search(clean(line.text) or ""))),
        None,
    )
    summary = _SOURCE_PREFIX.sub("", errors[-1]) if errors else None
    ended = exit_text(run.exit_code) if state == "exited" else None
    details: dict[str, Any] = {
        "state": state,
        "exit": ended,
        "family": family.name if family is not None else None,
        "detected_model": detected,
        "failed_while": stage,
        "files": {slot: _relative(path, models_dir) or path for slot, path in files.items()},
        "unusual_files": unusual,
        "errors": _last_unique(errors, ERROR_LINES),
        "warnings": _last_unique(warnings, WARNING_LINES),
        "log_tail": _collapse(t for _, t in lines)[-TAIL_LINES:],
        "hints": found,
        "log_file": str(run.log_path),
    }
    return EngineReport(
        state=state,
        summary=summary,
        exit=ended,
        exited_at=run.exited_at if state == "exited" else None,
        details={k: v for k, v in details.items() if v not in (None, [], {})},
    )
