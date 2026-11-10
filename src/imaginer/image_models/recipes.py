"""Image model families: what sd-server needs to run each one, and how to drive it.

A family is one of stable-diffusion.cpp's sd-server recipes (``data/sdcpp_recipes.json``, a
snapshot of ``examples/server/recipes``) with Stable-Imaginer's notes on it
(``data/families.yaml``). A recipe names the component slots a family loads (required, or
one of a group), its generation defaults, the multiple its sizes snap to and the features
it supports; the notes add how to group it, which LoRAs fit, how to prompt it, how to tell
its files apart and where on the Hub to get each component.

A running sd-server reports its own recipes (``GET /sdcpp/v1/recipes``); they replace the
snapshot's, so a family's slots and defaults match the engine build in use. Refresh the
snapshot with ``scripts/sync_sdcpp_recipes.py``.
"""

import json
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Literal

import yaml

from imaginer.core.logging import get_logger
from imaginer.domain.enums import Capability

log = get_logger(__name__)

# (slot, command-line flag, what it is) in launch order. Slot ids are sd-server's, as its
# recipes and POST /sdcpp/v1/models/load name them.
SLOTS: tuple[tuple[str, str, str], ...] = (
    ("model", "--model", "full model"),
    ("diffusion_model", "--diffusion-model", "diffusion model"),
    ("uncond_diffusion_model", "--uncond-diffusion-model", "unconditional diffusion model"),
    ("vae", "--vae", "VAE"),
    ("tae", "--tae", "tiny autoencoder"),
    ("clip_l", "--clip_l", "CLIP-L text encoder"),
    ("clip_g", "--clip_g", "CLIP-G text encoder"),
    ("t5xxl", "--t5xxl", "T5-XXL text encoder"),
    ("llm", "--llm", "text encoder"),
    ("llm_vision", "--llm_vision", "vision projector"),
    ("embeddings_connectors", "--embeddings-connectors", "embeddings connectors"),
    ("tokenizer", "--tokenizer", "tokenizer"),
)
SLOT_IDS = tuple(slot for slot, _, _ in SLOTS)
SLOT_WHAT = {slot: what for slot, _, what in SLOTS}
# The file a model's weights are named after: its checkpoint, else its diffusion model.
MAIN_SLOTS = ("model", "diffusion_model")
# Before families: an sd-server setup is these four (Qwen-Image-2.1's shape).
LEGACY_SLOTS = ("diffusion_model", "vae", "llm", "llm_vision")

# A recipe's generation defaults in our vocabulary (ImageDefaults, ImagePreset).
_DEFAULT_KEYS = {
    "sampling_method": "sampler",
    "cfg_scale": "cfg_scale",
    "steps": "steps",
    "width": "width",
    "height": "height",
    "scheduler": "scheduler",
    "flow_shift": "flow_shift",
    "custom_sigmas": "sigmas",
    "distilled_guidance": "guidance",
}
# Recipe defaults with no field of ours, sent as they are in sample_params.
_SAMPLE_KEYS = {"shifted_timestep": "shifted_timestep", "eta": "eta"}
OUR_DEFAULT_KEYS = frozenset({*_DEFAULT_KEYS.values(), "clip_skip", "negative_prompt"})

PromptStyle = Literal["natural", "tags", "json"]
Origin = Literal["builtin", "engine"]


@dataclass(frozen=True, slots=True)
class Slot:
    id: str
    required: bool
    label: str

    @property
    def what(self) -> str:
        return SLOT_WHAT.get(self.id, self.id)


@dataclass(frozen=True, slots=True)
class Family:
    id: str
    name: str
    description: str = ""
    model_url: str | None = None
    slots: tuple[Slot, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)  # ours; see _DEFAULT_KEYS
    extra_sample: Mapping[str, Any] = field(default_factory=dict)
    size_multiple: int = 8
    features: frozenset[str] = frozenset()  # the recipe's, less its disabled ones
    disabled: frozenset[str] = frozenset()
    vae_format: str | None = None
    model_args: Mapping[str, str] = field(default_factory=dict)
    origin: Origin = "builtin"
    # Our notes (families.yaml).
    hidden: bool = False
    group: str | None = None
    lora_group: str | None = None
    prompt: PromptStyle = "natural"
    hint: str | None = None
    note: str | None = None
    rgba: str | None = None  # the transparent-output template, with {prompt}
    mask: str | None = None  # how an edit's mask is explained in the prompt
    names: tuple[re.Pattern[str], ...] = ()
    repos: tuple[str, ...] = ()
    gguf: tuple[str, ...] = ()
    sources: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.slots)

    def slot(self, slot_id: str) -> Slot | None:
        return next((s for s in self.slots if s.id == slot_id), None)

    @property
    def main_slot(self) -> str:
        """Where the model's own weights go: ``model`` for a checkpoint family."""
        return next((s for s in MAIN_SLOTS if s in self.slot_ids), "diffusion_model")

    def missing(self, have: Iterable[str]) -> list[str]:
        """What a set of filled slots still lacks: each required slot, then each group
        with none of its slots filled ("vae or tae")."""
        filled = set(have)
        out = [s.id for s in self.slots if s.required and s.id not in filled]
        out += [" or ".join(group) for group in self.required_any if not filled & set(group)]
        return out

    def capabilities(self) -> set[Capability]:
        """What an engine running this family can do, at most. LoRA support still depends
        on the engine listing LoRAs."""
        caps = {Capability.T2I, Capability.LORA}
        if "ref_images" in self.features:
            caps |= {Capability.EDIT, Capability.MULTI_REF}
        if self.rgba:
            caps.add(Capability.RGBA)
        return caps

    @property
    def previews(self) -> bool:
        return "preview" not in self.disabled

    def prompt_hint(self) -> str | None:
        """What the chat model should know about prompting this family."""
        style = _HINTS.get(self.prompt) if self.prompt != "natural" else None
        parts = [p for p in (style, self.hint) if p]
        return " ".join(parts) or None

    def seed_defaults(self) -> dict[str, Any]:
        """Starting values for a new model card: the recipe's defaults, with the native
        size and defaults the notes give over them."""
        out = dict(self.defaults)
        if self.extra_sample:
            out["extra_sample"] = dict(self.extra_sample)
        return out


_HINTS: dict[str, str] = {}


def _family(
    recipe: Mapping[str, Any], notes: Mapping[str, Any], *, hidden: bool, origin: Origin
) -> Family:
    defaults_in = recipe.get("defaults") or {}
    defaults: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in defaults_in.items():
        if key in _DEFAULT_KEYS:
            defaults[_DEFAULT_KEYS[key]] = value
        elif key in _SAMPLE_KEYS:
            extra[_SAMPLE_KEYS[key]] = value
    native = notes.get("native")
    if isinstance(native, list) and len(native) == 2:
        defaults["width"], defaults["height"] = int(native[0]), int(native[1])
    for key, value in (notes.get("defaults") or {}).items():
        if key in OUR_DEFAULT_KEYS:
            defaults[key] = value
    if "sigmas" in defaults:
        defaults["sigmas"] = [float(s) for s in defaults["sigmas"]]
    context = recipe.get("context") or {}
    model_args = _parse_model_args(context.get("model_args"))
    model_args.update({str(k): arg_text(v) for k, v in (notes.get("model_args") or {}).items()})
    disabled = frozenset(str(f) for f in recipe.get("disabled_features") or [])
    slots = tuple(
        Slot(
            id=str(slot["id"]),
            required=bool(slot.get("required", False)),
            label=str(slot.get("label") or SLOT_WHAT.get(str(slot["id"]), slot["id"])),
        )
        for slot in recipe.get("slots") or []
        if isinstance(slot, Mapping) and slot.get("id")
    )
    prompt = notes.get("prompt", "natural")
    return Family(
        id=str(recipe["id"]),
        name=str(recipe.get("name") or recipe["id"]),
        description=str(recipe.get("description") or ""),
        model_url=recipe.get("model_url"),
        slots=slots,
        required_any=tuple(
            tuple(str(s) for s in group) for group in recipe.get("required_any") or []
        ),
        defaults=defaults,
        extra_sample=extra,
        size_multiple=int(recipe.get("size_multiple") or 8),
        features=frozenset(str(f) for f in recipe.get("features") or []) - disabled,
        disabled=disabled,
        vae_format=context.get("vae_format"),
        model_args=model_args,
        origin=origin,
        hidden=hidden,
        group=notes.get("group"),
        lora_group=notes.get("lora_group"),
        prompt=prompt if prompt in ("natural", "tags", "json") else "natural",
        hint=notes.get("hint"),
        note=notes.get("note"),
        rgba=notes.get("rgba"),
        mask=notes.get("mask"),
        names=tuple(re.compile(p, re.IGNORECASE) for p in notes.get("names") or []),
        repos=tuple(str(r) for r in notes.get("repos") or []),
        gguf=tuple(str(a) for a in notes.get("gguf") or []),
        sources={
            str(slot): tuple(str(ref) for ref in refs)
            for slot, refs in (notes.get("sources") or {}).items()
        },
    )


def arg_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_model_args(text: object) -> dict[str, str]:
    """``k=v,k=v`` (sd.cpp's --model-args) as a mapping."""
    out: dict[str, str] = {}
    if not isinstance(text, str):
        return out
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if key.strip():
            out[key.strip()] = value.strip() if sep else "true"
    return out


def model_args_text(args: Mapping[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in args.items())


class Catalog:
    """Every family, by recipe id: the snapshot's, replaced by an engine's own when one
    reports them."""

    def __init__(self, recipes: Iterable[Mapping[str, Any]], notes: Mapping[str, Any]) -> None:
        self._notes = dict(notes.get("families") or {})
        self._hidden = frozenset(str(h) for h in notes.get("hidden") or [])
        self._lock = threading.Lock()
        self._families: dict[str, Family] = {}
        for recipe in recipes:
            self._add(recipe, "builtin")

    def _add(self, recipe: Mapping[str, Any], origin: Origin) -> Family | None:
        if not isinstance(recipe, Mapping) or not recipe.get("id"):
            return None
        rid = str(recipe["id"])
        try:
            family = _family(
                recipe, self._notes.get(rid) or {}, hidden=rid in self._hidden, origin=origin
            )
        except (KeyError, TypeError, ValueError, re.error) as exc:
            log.warning("recipes.unreadable", recipe=rid, error=str(exc))
            return None
        with self._lock:
            self._families[rid] = family
        return family

    def get(self, family_id: str | None) -> Family | None:
        if family_id is None:
            return None
        with self._lock:
            return self._families.get(family_id)

    def families(self, *, hidden: bool = False) -> list[Family]:
        with self._lock:
            found = list(self._families.values())
        return sorted(
            (f for f in found if hidden or not f.hidden),
            key=lambda f: ((f.group or f.name).casefold(), f.name.casefold()),
        )

    def take_engine_recipes(self, recipes: object) -> int:
        """An engine's ``GET /sdcpp/v1/recipes`` ``recipes``: video ones are skipped, the
        rest replace the snapshot's by id. Returns how many were taken."""
        if not isinstance(recipes, list):
            return 0
        taken = 0
        for recipe in recipes:
            if not isinstance(recipe, Mapping) or _is_video(recipe):
                continue
            clean = {k: v for k, v in recipe.items() if k not in _RESPONSE_ONLY}
            current = self.get(str(clean.get("id")))
            if current is not None and current.origin == "engine":
                continue  # the first engine to report it decides
            if self._add(clean, "engine") is not None:
                taken += 1
        return taken


# Fields sd-server adds to the recipes it reports (api.md), not part of a recipe.
_RESPONSE_ONLY = frozenset({"source", "file", "effective_features"})
_VIDEO_SLOTS = frozenset(
    {"high_noise_diffusion_model", "audio_vae", "audio_encoder", "motion_module"}
)
_VIDEO_FEATURES = frozenset({"ref_videos", "ref_audios", "ref_video_audios", "high_noise"})


def _is_video(recipe: Mapping[str, Any]) -> bool:
    if "video_constraints" in recipe:
        return True
    if any(str(k).startswith("video") for k in recipe.get("defaults") or {}):
        return True
    slots = {s.get("id") for s in recipe.get("slots") or [] if isinstance(s, Mapping)}
    return bool(slots & _VIDEO_SLOTS or set(recipe.get("features") or []) & _VIDEO_FEATURES)


_CATALOG: Catalog | None = None
_CATALOG_LOCK = threading.Lock()


def catalog() -> Catalog:
    """The process-wide catalog, loaded from the package data on first use."""
    global _CATALOG
    with _CATALOG_LOCK:
        if _CATALOG is None:
            data = resources.files("imaginer.image_models").joinpath("data")
            snapshot = json.loads(data.joinpath("sdcpp_recipes.json").read_text(encoding="utf-8"))
            notes = yaml.safe_load(data.joinpath("families.yaml").read_text(encoding="utf-8"))
            _HINTS.clear()
            _HINTS.update(
                {str(k): " ".join(str(v).split()) for k, v in (notes.get("hints") or {}).items()}
            )
            _CATALOG = Catalog(snapshot.get("recipes") or [], notes or {})
        return _CATALOG


def family(family_id: str | None) -> Family | None:
    return catalog().get(family_id)
