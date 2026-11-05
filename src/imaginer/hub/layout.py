"""What a Hub repo holds, sorted into what the engines load.

A pure function of the repo's file list (plus ``model_index.json`` for Diffusers repos), so
it runs the same on a live listing and in tests. The shapes it knows:

* **GGUF:** ``<name>-<QUANT>.gguf`` side by side (``Q4_K_M``, ``UD-Q4_K_XL``, ``BF16``...), or
  one folder per quant for big models, split into ``-00001-of-00003`` shards. Vision
  projectors are ``mmproj-*.gguf``; ``imatrix`` files are calibration data, not models.
* **Diffusers:** ``model_index.json`` names the pipeline class and a folder per component
  (``transformer/``, ``vae/``, ``text_encoder/``...), each with (often sharded) safetensors.
  A diffusers backend loads the whole pipeline; sd-server loads a component on its own,
  through its ``*.safetensors.index.json`` when it is sharded.
* **ComfyUI repackages:** single files under ``diffusion_models/``, ``vae/`` and
  ``text_encoders/`` (sometimes inside ``split_files/``), which sd-server loads directly.
* **transformers:** ``config.json`` and ``model-*.safetensors`` shards. llama.cpp needs a GGUF
  conversion, which the Hub lists as the repo's quantized versions; sd-server can read one as
  a text encoder through its index.
* **Loose safetensors:** anything else, classified by folder and file name.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

LayoutKind = Literal["gguf", "diffusers", "comfy", "transformers", "safetensors", "unknown"]
Component = Literal["model", "mmproj", "diffusion_model", "vae", "text_encoder", "lora", "other"]
WeightFormat = Literal["gguf", "safetensors"]
TaskKind = Literal["image", "llm"]

IMAGE_TASKS = frozenset(
    {"text-to-image", "image-to-image", "unconditional-image-generation", "image-text-to-image"}
)
LLM_TASKS = frozenset(
    {"text-generation", "image-text-to-text", "text2text-generation", "visual-question-answering"}
)
WEIGHTS: dict[str, WeightFormat] = {".gguf": "gguf", ".safetensors": "safetensors"}
PICKLED = frozenset({".bin", ".ckpt", ".pt", ".pth"})
# Weight formats a Diffusers folder may carry alongside safetensors; from_pretrained
# prefers the safetensors, so the others are not downloaded.
OTHER_WEIGHTS = PICKLED | {".msgpack", ".h5", ".onnx", ".onnx_data", ".pb", ".ot"}

_ShardKey = tuple[str, str, int, WeightFormat]  # folder, base name, shard count, format
_SHARD = re.compile(r"^(?P<base>.+)-(?P<n>\d{5})-of-(?P<total>\d{5})$")
_QUANT = re.compile(
    r"(?:^|[-_.])(?P<q>(?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|TQ\d_\d|MXFP4(?:_MOE)?|NVFP4"
    r"|BF16|FP16|F16|FP32|F32|FP8(?:_E4M3FN|_E5M2)?(?:_SCALED)?|INT8(?:[-_]CONVROT)?|INT4"
    r"|W4A8|W8A8))$",
    re.IGNORECASE,
)
# A Diffusers weight variant (``diffusion_pytorch_model.fp16.safetensors``), loaded only
# when asked for by name.
_DIFFUSERS_VARIANT = re.compile(r"\.(?:fp16|bf16|fp32|non_ema|ema)(?:[.-]|$)")

# The innermost folder that names a component decides it.
_FOLDERS: dict[str, Component] = {
    "diffusion_models": "diffusion_model",
    "unet": "diffusion_model",
    "transformer": "diffusion_model",
    "checkpoints": "diffusion_model",
    "vae": "vae",
    "text_encoders": "text_encoder",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "clip": "text_encoder",
    "loras": "lora",
    "lora": "lora",
    "clip_vision": "other",
    "model_patches": "other",
    "controlnet": "other",
    "upscale_models": "other",
    "embeddings": "other",
}
# Folder names only ComfyUI's model tree uses.
_COMFY = frozenset(
    {"diffusion_models", "text_encoders", "unet", "checkpoints", "loras", "clip_vision"}
)
_CHECKPOINT_NAMES = frozenset({"model", "pytorch_model", "diffusion_pytorch_model"})
_TEXT_ENCODER_WORDS = ("text_encoder", "text-encoder", "textencoder")
_TEXT_ENCODER_TOKENS = frozenset({"t5", "t5xxl", "umt5", "clip"})

# Which quant to suggest when nothing more specific is known: a good size/quality balance
# first, then larger, then smaller.
PREFERRED_QUANTS = (
    "Q4_K_M", "Q4_K", "UD-Q4_K_XL", "Q4_K_S", "Q5_K_M", "Q5_K", "UD-Q5_K_XL", "Q5_K_S",
    "Q4_0", "IQ4_XS", "IQ4_NL", "Q6_K", "UD-Q6_K_XL", "Q8_0", "BF16", "FP16", "F16",
    "FP8_E4M3FN", "FP8", "FP8_SCALED", "INT8", "INT8_CONVROT", "W8A8", "Q3_K_M", "Q3_K",
    "UD-Q3_K_XL", "Q3_K_S", "W4A8", "NVFP4", "Q2_K", "Q2_K_L", "F32", "FP32",
)  # fmt: skip
PREFERRED_PROJECTORS = ("F16", "BF16", "Q8_0", "F32", "FP16", "FP32")


@dataclass(frozen=True, slots=True)
class RepoFile:
    path: str
    size: int | None = None
    sha256: str | None = None  # LFS files: what a download is checked against


@dataclass(frozen=True, slots=True)
class Variant:
    """One loadable model: a single file, or the shards of one.

    ``files`` is everything to download, first the file an engine's flag points at: the file
    itself, a split GGUF's first shard, or sharded safetensors' ``.safetensors.index.json``
    (sd-server reads the index to find the shards)."""

    name: str
    component: Component
    format: WeightFormat
    files: tuple[RepoFile, ...]
    quant: str | None = None

    @property
    def id(self) -> str:
        """The path an engine's flag points at."""
        return self.files[0].path

    @property
    def indexed(self) -> bool:
        return self.id.endswith(".json")

    @property
    def folder(self) -> str:
        parent = str(PurePosixPath(self.id).parent)
        return "" if parent == "." else parent

    @property
    def size(self) -> int:
        return sum(f.size or 0 for f in self.files)

    @property
    def sharded(self) -> bool:
        return len(self.files) - self.indexed > 1


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A Diffusers pipeline: everything ``from_pretrained`` reads from the repo."""

    pipeline_class: str | None
    components: tuple[str, ...]
    files: tuple[RepoFile, ...]
    weights_only_as_variant: bool = False  # e.g. only *.fp16.safetensors: needs variant=

    @property
    def size(self) -> int:
        return sum(f.size or 0 for f in self.files)


@dataclass(frozen=True, slots=True)
class RepoLayout:
    kind: LayoutKind
    variants: tuple[Variant, ...]
    pipeline: Pipeline | None = None
    task: TaskKind | None = None
    notes: tuple[str, ...] = ()

    def variant(self, variant_id: str) -> Variant | None:
        return next((v for v in self.variants if v.id == variant_id), None)


def task_kind(pipeline_tag: str | None) -> TaskKind | None:
    if pipeline_tag in IMAGE_TASKS:
        return "image"
    if pipeline_tag in LLM_TASKS:
        return "llm"
    return None


def classify(
    files: Iterable[RepoFile],
    *,
    repo_id: str = "",
    pipeline_tag: str | None = None,
    model_index: Mapping[str, Any] | None = None,
) -> RepoLayout:
    files = list(files)
    paths = {f.path for f in files}
    task = task_kind(pipeline_tag)
    notes: list[str] = []

    pipeline = None
    if "model_index.json" in paths:
        pipeline = _pipeline(files, model_index)
        if pipeline.weights_only_as_variant:
            notes.append(
                "The pipeline's weights are only published as a Diffusers variant "
                "(e.g. fp16), which this server doesn't load by name."
            )
    components = set(pipeline.components) if pipeline else set()

    groups: dict[_ShardKey, list[tuple[int, RepoFile]]] = defaultdict(list)
    for f in files:
        p = PurePosixPath(f.path)
        fmt = WEIGHTS.get(p.suffix.lower())
        if fmt is None:
            continue
        if fmt == "gguf" and "imatrix" in p.name.lower():
            continue
        stem = p.name[: -len(p.suffix)]
        shard = _SHARD.match(stem)
        if shard:
            base, n, total = shard["base"], int(shard["n"]), int(shard["total"])
        else:
            base, n, total = stem, 1, 1
        groups[(str(p.parent), base, total, fmt)].append((n, f))

    by_path = {f.path: f for f in files}
    repo_name = repo_id.rpartition("/")[2]
    variants: list[Variant] = []
    for (folder, base, total, fmt), members in groups.items():
        members.sort(key=lambda m: m[0])
        if [n for n, _ in members] != list(range(1, total + 1)):
            notes.append(f"{base}: only {len(members)} of its {total} shards are here; skipped.")
            continue
        shards = tuple(f for _, f in members)
        prefix = "" if folder == "." else f"{folder}/"
        index = by_path.get(f"{prefix}{base}.safetensors.index.json")
        if fmt == "safetensors" and total > 1 and index is not None:
            shards = (index, *shards)
        # Weights named for their library rather than their model are named by their folder.
        generic = base.lower() in _CHECKPOINT_NAMES
        variants.append(
            Variant(
                name=(prefix.rstrip("/") or repo_name or base) if generic else base,
                component=_component(
                    shards[-1].path, base, task, repo_id, "config.json" in paths, components
                ),
                format=fmt,
                files=shards,
                quant=quant_of(base),
            )
        )
    variants.sort(key=lambda v: (v.folder, v.component, v.name.lower()))

    kind = _kind(files, pipeline)
    if kind == "transformers":
        notes.append(
            "A transformers checkpoint: llama.cpp needs a GGUF conversion of it (see this "
            "repo's quantized versions). sd-server reads it as a text encoder, for the "
            "models it supports, through its index."
        )
    if not variants and pipeline is None:
        if any(PurePosixPath(f.path).suffix.lower() in PICKLED for f in files):
            notes.append("Only pickled weights (.bin, .ckpt, .pt): nothing here loads them.")
        else:
            notes.append("No GGUF or safetensors weights in this repo.")
    return RepoLayout(kind, tuple(variants), pipeline, task, tuple(notes))


def shard_base(file_name: str) -> str:
    """A weight file's name without its extension and any ``-00001-of-00003`` suffix."""
    p = PurePosixPath(file_name)
    stem = p.name[: -len(p.suffix)] if p.suffix.lower() in WEIGHTS else p.name
    shard = _SHARD.match(stem)
    return shard["base"] if shard else stem


def quant_of(name: str) -> str | None:
    match = _QUANT.search(name)
    if match is None:
        return None
    quant = match["q"].upper()
    return quant if quant.startswith("UD-") else quant.replace("-", "_")


def _component(
    path: str,
    base: str,
    task: TaskKind | None,
    repo_id: str,
    configured: bool,
    pipeline_components: set[str],
) -> Component:
    p = PurePosixPath(path)
    lower = base.lower()
    if "mmproj" in lower:
        return "mmproj"
    for folder in reversed(p.parts[:-1]):
        found = _FOLDERS.get(folder.lower())
        if found is not None:
            return found
    if len(p.parts) > 1 and p.parts[0] in pipeline_components:
        return "other"  # a pipeline part no engine setting takes (image_encoder, ...)
    repo_name = repo_id.rpartition("/")[2].lower()
    if lower == "diffusion_pytorch_model":
        # One Diffusers component published on its own: only the repo's name says which.
        repo_tokens = set(re.split(r"[-_.\s]+", repo_name))
        if "vae" in repo_tokens:
            return "vae"
        if repo_tokens & {"transformer", "unet", "dit"}:
            return "diffusion_model"
        return "other"
    tokens = set(re.split(r"[-_.\s]+", lower))
    if "vae" in tokens or lower in {"ae", "vae"}:
        return "vae"
    if "lora" in tokens or "lora" in lower:
        return "lora"
    if any(w in lower for w in _TEXT_ENCODER_WORDS) or tokens & _TEXT_ENCODER_TOKENS:
        return "text_encoder"
    if "controlnet" in lower:
        return "other"
    if any(w in repo_name for w in _TEXT_ENCODER_WORDS):
        return "text_encoder"
    if configured and lower in _CHECKPOINT_NAMES:
        return "model"  # a transformers checkpoint
    return "diffusion_model" if task == "image" else "model"


def _kind(files: list[RepoFile], pipeline: Pipeline | None) -> LayoutKind:
    if pipeline is not None:
        return "diffusers"
    weights = [p for p in (PurePosixPath(f.path) for f in files) if p.suffix.lower() in WEIGHTS]
    if any(part.lower() in _COMFY for p in weights for part in p.parts[:-1]):
        return "comfy"
    if any(p.suffix.lower() == ".gguf" and "imatrix" not in p.name.lower() for p in weights):
        return "gguf"
    root = [p.name for p in weights if len(p.parts) == 1]
    if any(f.path == "config.json" for f in files) and any(
        shard_base(name) in {"model", "pytorch_model"} for name in root
    ):
        return "transformers"
    if weights:
        return "safetensors"
    return "unknown"


def _pipeline(files: list[RepoFile], model_index: Mapping[str, Any] | None) -> Pipeline:
    by_folder: dict[str, list[RepoFile]] = defaultdict(list)
    for f in files:
        top, sep, _ = f.path.partition("/")
        if sep:
            by_folder[top].append(f)
    if model_index is not None:
        # Each component is [library, class]; [null, null] marks an optional one left out.
        components = [
            key
            for key, value in model_index.items()
            if not key.startswith("_")
            and isinstance(value, list | tuple)
            and len(value) == 2
            and value[0] is not None
            and key in by_folder
        ]
        pipeline_class = model_index.get("_class_name")
    else:
        components = [
            folder
            for folder, members in by_folder.items()
            if any(PurePosixPath(f.path).name.endswith("config.json") for f in members)
        ]
        pipeline_class = None
    kept: list[RepoFile] = [f for f in files if f.path == "model_index.json"]
    variant_only = False
    for folder in sorted(components):
        members = by_folder[folder]
        weights = [f for f in members if PurePosixPath(f.path).suffix.lower() in WEIGHTS]
        plain = [f for f in weights if not _DIFFUSERS_VARIANT.search(PurePosixPath(f.path).name)]
        variant_only = variant_only or bool(weights and not plain)
        has_safetensors = any(f.path.endswith(".safetensors") for f in members)
        for f in members:
            name = PurePosixPath(f.path).name
            suffix = PurePosixPath(name).suffix.lower()
            if has_safetensors and suffix in OTHER_WEIGHTS:
                continue
            if plain and _DIFFUSERS_VARIANT.search(name):
                continue
            kept.append(f)
    return Pipeline(
        pipeline_class if isinstance(pipeline_class, str) else None,
        tuple(sorted(components)),
        tuple(kept),
        variant_only,
    )


# --- fitting variants to an engine's launch settings ----------------------------------


@dataclass(frozen=True, slots=True)
class Role:
    """What an engine's launch setting can take from a repo."""

    components: frozenset[Component]
    formats: frozenset[WeightFormat]
    split_gguf: bool  # llama.cpp opens a split GGUF from its first shard; sd-server doesn't
    indexed: bool = False  # sd-server reads sharded safetensors through their index

    def loads(self, variant: Variant) -> bool:
        if variant.format not in self.formats:
            return False
        if not variant.sharded:
            return True
        return self.split_gguf if variant.format == "gguf" else self.indexed and variant.indexed


_GGUF: frozenset[WeightFormat] = frozenset({"gguf"})
_ANY: frozenset[WeightFormat] = frozenset({"gguf", "safetensors"})
ROLES: dict[str, dict[str, Role]] = {
    "llamacpp": {
        "model": Role(frozenset({"model", "text_encoder"}), _GGUF, split_gguf=True),
        "mmproj": Role(frozenset({"mmproj"}), _GGUF, split_gguf=False),
    },
    "sdcpp": {
        "diffusion_model": Role(frozenset({"diffusion_model"}), _ANY, False, indexed=True),
        "vae": Role(frozenset({"vae"}), _ANY, False, indexed=True),
        "llm": Role(frozenset({"text_encoder", "model"}), _ANY, False, indexed=True),
        "llm_vision": Role(frozenset({"mmproj"}), _GGUF, split_gguf=False),
    },
}


def roles_for(engine_type: str, layout: RepoLayout) -> dict[str, list[Variant]]:
    """Launch setting -> the variants in this repo that can fill it."""
    out: dict[str, list[Variant]] = {}
    for key, role in ROLES.get(engine_type, {}).items():
        components = role.components
        if engine_type == "sdcpp" and key == "diffusion_model" and layout.task is None:
            components = components | {"model"}  # an untagged repo's models may be diffusion ones
        out[key] = [v for v in layout.variants if v.component in components and role.loads(v)]
    return out


def ranked(candidates: list[Variant], *, key: str) -> list[Variant]:
    """Most useful first: by quant preference, then fewer files, smaller, plainer name."""
    order = PREFERRED_PROJECTORS if key in {"mmproj", "llm_vision"} else PREFERRED_QUANTS

    def rank(v: Variant) -> tuple[int, int, int, int]:
        position = order.index(v.quant) if v.quant in order else len(order)
        return (position, len(v.files), v.size, len(v.id))

    return sorted(candidates, key=rank)


def suggest(candidates: list[Variant], *, key: str, quant: str | None = None) -> Variant | None:
    """The variant to offer first: the asked-for quant, else the preferred one."""
    if quant is not None:
        wanted = quant.upper()
        exact = [v for v in candidates if v.quant == wanted]
        if exact:
            return min(exact, key=lambda v: v.size)
    best = ranked(candidates, key=key)
    return best[0] if best else None
