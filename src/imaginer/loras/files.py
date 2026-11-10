"""What a LoRA's file and a model's weights say about each other, read from their headers
only: a safetensors file's JSON header, a GGUF file's tensor table. Nothing here loads
weights.

A LoRA patches named modules of the model it was trained on. Trainers write those names
in several conventions (Diffusers/PEFT ``transformer.blocks.0.attn.to_q.lora_A.weight``,
kohya ``lora_unet_blocks_0_attn_to_q.lora_down.weight``, LyCORIS), and model files carry
prefixes of their own (``model.diffusion_model.``), so both sides are reduced to one
canonical form before they are compared, together with each module's input width. The
share of a LoRA's modules a model has, at the same width, is a good first guess at whether
it applies there: Qwen-Image 2.1 and an older Qwen-Image share many module names, but not
their width. It can't tell two checkpoints of one architecture apart (a base model and its
fine-tune have the same modules); the user's call settles that.

Everything is remembered per file while its size and modification time stay the same.
Blocking: call from a worker thread.
"""

import json
import re
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

HEADER_MAX = 64 * 1024 * 1024  # a safetensors header larger than this isn't trusted
GGUF_MAX_KV = 100_000
GGUF_MAX_TENSORS = 1_000_000
GGUF_MAX_STRING = 16 * 1024 * 1024
WORDS_MAX = 12
WORD_MAX = 200

# Metadata keys trainers write trigger words under. A phrase stays whole; a list of words
# (or a comma-separated one) becomes several.
PHRASE_KEYS = ("modelspec.trigger_phrase", "trigger_phrase")
WORD_KEYS = ("trigger_word", "trigger_words", "instance_prompt")
# Metadata keys that name the exact model a LoRA was made for, most specific first.
BASE_KEYS = ("base_model", "modelspec.base_model", "ss_sd_model_name", "ss_base_model")
# Ones that only name its family ("sdxl_base_v1-0", "stable-diffusion-xl-v1-base/lora").
FAMILY_KEYS = ("ss_base_model_version", "modelspec.architecture")

# The tensor of each LoRA module whose second dimension is the module's input width.
_DOWN_SUFFIXES = (
    ".lora_A.default.weight", ".lora_A.weight", ".lora_down.weight", ".lora.down.weight",
)  # fmt: skip
_LORA_SUFFIXES = (
    *_DOWN_SUFFIXES, ".lora_B.default.weight", ".lora_B.weight", ".lora_up.weight",
    ".lora.up.weight", ".lora_mid.weight", ".hada_w1_a", ".hada_w1_b", ".hada_w2_a",
    ".hada_w2_b", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a",
    ".lokr_w2_b", ".alpha", ".dora_scale", ".diff", ".diff_b",
)  # fmt: skip
_PREFIXES = (
    "base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.",
    "unet.", "lora_unet_", "lora_transformer_", "lycoris_", "model.",
)  # fmt: skip
# Layers a model file may store fused, by the names a LoRA patches them under.
_FUSED = {"_gate_up": ("_gate_layer", "_proj")}
_CACHE: dict[tuple[str, str, int, int], Any] = {}
_CACHE_MAX = 512

Modules = Mapping[str, int | None]  # canonical module name -> its input width, if known


def _cached(kind: str, path: Path, read: Any) -> Any:
    stat = path.stat()
    key = (kind, str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = read(path)
    return _CACHE[key]


# --- safetensors ---------------------------------------------------------------------------


def safetensors_header(path: Path) -> dict[str, Any] | None:
    """The JSON header of a .safetensors file (tensor names to dtype/shape, and
    ``__metadata__``), or None if it isn't one."""
    try:
        return _cached("st", path, _read_safetensors)  # type: ignore[no-any-return]
    except OSError:
        return None


def _read_safetensors(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as fh:
            (size,) = struct.unpack("<Q", fh.read(8))
            if size > HEADER_MAX:
                return None
            header = json.loads(fh.read(size))
    except (OSError, ValueError, struct.error):
        return None
    return header if isinstance(header, dict) else None


def _tensors(header: dict[str, Any]) -> list[tuple[str, list[int] | None]]:
    """(name, shape) of each tensor in a safetensors header, outermost dimension first."""
    out = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        shape = spec.get("shape") if isinstance(spec, dict) else None
        ok = isinstance(shape, list) and all(isinstance(d, int) for d in shape)
        out.append((name, shape if ok else None))
    return out


# --- GGUF ----------------------------------------------------------------------------------

_GGUF_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class _Bad(ValueError):
    pass


def _u32(fh: BinaryIO) -> int:
    data = fh.read(4)
    if len(data) != 4:
        raise _Bad("short read")
    return int(struct.unpack("<I", data)[0])


def _u64(fh: BinaryIO) -> int:
    data = fh.read(8)
    if len(data) != 8:
        raise _Bad("short read")
    return int(struct.unpack("<Q", data)[0])


def _string(fh: BinaryIO) -> str:
    n = _u64(fh)
    if n > GGUF_MAX_STRING:
        raise _Bad("string too long")
    data = fh.read(n)
    if len(data) != n:
        raise _Bad("short read")
    return data.decode("utf-8", "replace")


def _skip_value(fh: BinaryIO, kind: int) -> None:
    if kind in _GGUF_SIZES:
        fh.seek(_GGUF_SIZES[kind], 1)
    elif kind == _GGUF_STRING:
        fh.seek(_u64(fh), 1)
    elif kind == _GGUF_ARRAY:
        item, count = _u32(fh), _u64(fh)
        if item in _GGUF_SIZES:
            fh.seek(_GGUF_SIZES[item] * count, 1)
        else:
            for _ in range(count):
                _skip_value(fh, item)
    else:
        raise _Bad(f"unknown GGUF value type {kind}")


def gguf_tensors(path: Path) -> list[tuple[str, list[int]]] | None:
    """(name, shape) of each tensor of a GGUF file (version 2 or later), from its header,
    outermost dimension first like safetensors (GGUF stores them innermost first)."""
    try:
        return _cached("gguf", path, _read_gguf)  # type: ignore[no-any-return]
    except OSError:
        return None


def _read_gguf(path: Path) -> list[tuple[str, list[int]]] | None:
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF" or _u32(fh) < 2:
                return None
            n_tensors, n_kv = _u64(fh), _u64(fh)
            if n_tensors > GGUF_MAX_TENSORS or n_kv > GGUF_MAX_KV:
                return None
            for _ in range(n_kv):
                _string(fh)
                _skip_value(fh, _u32(fh))
            tensors = []
            for _ in range(n_tensors):
                name = _string(fh)
                dims = _u32(fh)
                if dims > 8:
                    return None
                shape = [_u64(fh) for _ in range(dims)]
                fh.seek(4 + 8, 1)  # type, offset
                tensors.append((name, shape[::-1]))
            return tensors
    except (OSError, _Bad, struct.error):
        return None


def gguf_metadata(path: Path) -> dict[str, str] | None:
    """A GGUF file's string metadata (``general.architecture``, ``general.name``...); None
    if it isn't a GGUF file."""
    try:
        return _cached("gguf-meta", path, _read_gguf_meta)  # type: ignore[no-any-return]
    except OSError:
        return None


def _read_gguf_meta(path: Path) -> dict[str, str] | None:
    try:
        with path.open("rb") as fh:
            if fh.read(4) != b"GGUF" or _u32(fh) < 2:
                return None
            _u64(fh)
            n_kv = _u64(fh)
            if n_kv > GGUF_MAX_KV:
                return None
            meta = {}
            for _ in range(n_kv):
                key = _string(fh)
                kind = _u32(fh)
                if kind == _GGUF_STRING:
                    meta[key] = _string(fh)
                else:
                    _skip_value(fh, kind)
            return meta
    except (OSError, _Bad, struct.error):
        return None


def tensor_names(path: Path) -> list[str] | None:
    """The tensor names of a model file: a GGUF or safetensors file, or a safetensors index
    (``*.safetensors.index.json``, or a folder holding ``model.safetensors.index.json``);
    None if they can't be read."""
    try:
        if path.is_dir():
            path = path / "model.safetensors.index.json"
        if path.name.endswith(".json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            return list(weight_map) if isinstance(weight_map, dict) else None
        if path.suffix.lower() == ".gguf":
            tensors = gguf_tensors(path)
            return [name for name, _ in tensors] if tensors is not None else None
        header = safetensors_header(path)
    except (OSError, ValueError):
        return None
    return [k for k in header if k != "__metadata__"] if header is not None else None


# --- module names --------------------------------------------------------------------------


def canonical(name: str) -> str:
    """A module name with the prefixes and separators of each convention taken out."""
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name, changed = name[len(prefix) :], True
    return name.replace(".", "_").lower()


def _width(shape: list[int] | None) -> int | None:
    """A weight's input width: the second dimension of [out, in, ...]."""
    return shape[1] if shape is not None and len(shape) >= 2 else None


def lora_modules(tensors: Iterable[tuple[str, list[int] | None]]) -> dict[str, int | None]:
    """The model modules a LoRA's tensors patch, each with its input width if it says."""
    found: dict[str, int | None] = {}
    for key, shape in tensors:
        for suffix in _LORA_SUFFIXES:
            if key.endswith(suffix):
                module = canonical(key[: -len(suffix)])
                width = _width(shape) if suffix in _DOWN_SUFFIXES else None
                found[module] = width if width is not None else found.get(module)
                break
    return found


def model_modules(tensors: Iterable[tuple[str, list[int] | None]]) -> dict[str, int | None]:
    """A model's weighted modules and their input widths, from its tensors. A fused layer
    also counts under the names a LoRA patches its parts by."""
    found: dict[str, int | None] = {}
    for name, shape in tensors:
        if not name.endswith(".weight"):
            continue
        module = canonical(name[: -len(".weight")])
        found[module] = _width(shape)
        for fused, parts in _FUSED.items():
            if module.endswith(fused):
                for part in parts:
                    found.setdefault(module[: -len(fused)] + part, found[module])
    return found


def matched(lora: Modules, model: Modules) -> int:
    """How many of a LoRA's modules the model has, at the same input width when both say.
    A name that isn't found as it is gets a second try without its first few parts, for a
    trainer's prefix this module doesn't know."""
    hits = 0
    for module, width in lora.items():
        names = [module]
        parts = module.split("_")
        names += ["_".join(parts[i:]) for i in range(1, min(4, len(parts)))]
        for name in names:
            if name in model:
                theirs = model[name]
                if width is None or theirs is None or width == theirs:
                    hits += 1
                break
    return hits


# --- a LoRA's facts ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoraFacts:
    modules: Modules = field(default_factory=dict)  # empty when its keys can't be read
    format: str | None = None  # "PEFT keys", "kohya keys", "LyCORIS keys"
    rank: int | None = None
    base: str | None = None  # the model it says it was made for
    family: str | None = None  # the model family it says it is for
    words: tuple[str, ...] = ()  # trigger words its metadata names
    words_from: str | None = None  # the metadata key they came from


NO_FACTS = LoraFacts()


def lora_facts(path: Path) -> LoraFacts:
    """What a LoRA file says about itself. Only .safetensors and .gguf headers are read."""
    try:
        return _cached("facts", path, _read_facts)  # type: ignore[no-any-return]
    except OSError:
        return NO_FACTS


def _read_facts(path: Path) -> LoraFacts:
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        tensors: list[tuple[str, list[int] | None]] = list(_read_gguf(path) or [])
        return LoraFacts(modules=lora_modules(tensors), format=_format_of(n for n, _ in tensors))
    if suffix not in {".safetensors", ".sft"}:
        return NO_FACTS
    header = _read_safetensors(path)
    if header is None:
        return NO_FACTS
    raw = header.get("__metadata__")
    meta = {k: v for k, v in raw.items() if isinstance(v, str)} if isinstance(raw, dict) else {}
    tensors = _tensors(header)
    rank = next(
        (min(s) for n, s in tensors if s and n.endswith(_DOWN_SUFFIXES) and len(s) >= 2), None
    )
    words, words_from = metadata_words(meta)
    return LoraFacts(
        modules=lora_modules(tensors),
        format=_format_of(n for n, _ in tensors),
        rank=rank,
        base=_first(meta, BASE_KEYS),
        family=_first(meta, FAMILY_KEYS),
        words=words,
        words_from=words_from,
    )


def _format_of(keys: Iterable[str]) -> str | None:
    seen = set()
    for key in keys:
        if ".lora_A." in key or ".lora_B." in key:
            seen.add("PEFT keys")
        elif ".lora_down." in key or ".lora_up." in key or ".lora.down." in key:
            seen.add("kohya keys")
        elif ".hada_" in key or ".lokr_" in key:
            seen.add("LyCORIS keys")
    return " + ".join(sorted(seen)) or None


def _first(meta: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = meta.get(key, "").strip()
        if value and value.lower() not in {"none", "null", "unknown"}:
            return value[:200]
    return None


def metadata_words(meta: dict[str, str]) -> tuple[tuple[str, ...], str | None]:
    """The trigger words a LoRA's metadata names, and the key they came from."""
    for key in (*PHRASE_KEYS, *WORD_KEYS):
        value = meta.get(key, "").strip()
        if not value:
            continue
        if value.startswith("["):  # some trainers store a JSON list of words
            try:
                items = json.loads(value)
            except ValueError:
                items = None
            if isinstance(items, list):
                return split_words(", ".join(str(i) for i in items)), key
        return split_words(value, phrase=key in PHRASE_KEYS), key
    return (), None


def split_words(text: str, *, phrase: bool = False) -> tuple[str, ...]:
    """Trigger words from text: one phrase, or several separated by commas."""
    parts = [text] if phrase else text.split(",")
    out: list[str] = []
    for part in parts:
        word = " ".join(part.split())[:WORD_MAX]
        if word and word.casefold() not in {w.casefold() for w in out}:
            out.append(word)
    return tuple(out[:WORDS_MAX])


# --- a model's modules ---------------------------------------------------------------------

# Where a Diffusers pipeline keeps the network LoRAs patch.
_DENOISERS = ("transformer", "unet")


def model_file_modules(path: Path) -> Modules | None:
    """A model file's modules (GGUF or safetensors); None if they can't be read."""
    try:
        return _cached("model", path, _read_model_file)  # type: ignore[no-any-return]
    except OSError:
        return None


def _read_model_file(path: Path) -> Modules | None:
    if path.suffix.lower() == ".gguf":
        tensors: list[tuple[str, list[int] | None]] | None = list(_read_gguf(path) or []) or None
    else:
        header = _read_safetensors(path)
        tensors = _tensors(header) if header else None
    return model_modules(tensors) if tensors else None


def pipeline_modules(folder: Path) -> Modules | None:
    """A Diffusers pipeline folder's denoiser modules, from the headers of its weight
    files; None if it has none (not downloaded yet)."""
    for sub in _DENOISERS:
        part = folder / sub
        if not part.is_dir():
            continue
        tensors: list[tuple[str, list[int] | None]] = []
        for file in sorted(part.glob("*.safetensors")):
            header = safetensors_header(file)
            if header:
                tensors += _tensors(header)
        if tensors:
            return model_modules(tensors)
    return None


# --- model names ---------------------------------------------------------------------------

_EXTENSION = re.compile(r"\.(gguf|safetensors|sft|ckpt|pt|pth|bin)$", re.IGNORECASE)
_SHARD = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_QUANT = re.compile(
    r"[-_.]?(?:UD[-_.])?(I?Q\d\w*|F16|BF16|FP16|F32|FP32|FP8\w*|INT8|NF4)$", re.IGNORECASE
)


_SNAPSHOT = re.compile(r"/models--[^/]+--([^/]+)/snapshots/[0-9a-f]{40}$", re.IGNORECASE)


def _stem(text: str) -> str:
    path = text.replace("\\", "/").rstrip("/")
    cached = _SNAPSHOT.search(path)  # a Hugging Face cache snapshot: its repo's name
    name = cached.group(1) if cached else path.rsplit("/", 1)[-1]
    return _SHARD.sub("", _EXTENSION.sub("", name))


def model_name(text: str) -> str:
    """A model as people name it: "D:/m/Qwen-Image-2.1-Q4_K_M.gguf" and "Qwen/Qwen-Image-2.1"
    are both "Qwen-Image-2.1"."""
    stem = _stem(text)
    return _QUANT.sub("", stem) or stem or text


def model_quant(text: str) -> str | None:
    """The quantization in a model file's name ("Q4_K_M"), if it says."""
    found = _QUANT.search(_stem(text))
    return found.group(1).upper() if found else None


def model_key(name: str) -> str:
    return model_name(name).casefold()


def lineage_matches(lineage: str, name: str) -> bool:
    """Whether a model's name is in a lineage: "Qwen-Image 2.x" takes Qwen-Image-2.1 and
    Qwen-Image-2.1-Turbo, but not Qwen-Image 1.0. "x" or "*" stands for any version."""
    tokens = re.findall(r"[a-z0-9]+|\*", lineage.casefold())
    if not tokens:
        return False
    parts = ["[a-z0-9]*" if t in {"x", "*"} else re.escape(t) for t in tokens]
    return re.search(r"[\W_]*".join(parts), name.casefold()) is not None
