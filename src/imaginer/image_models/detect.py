"""Which family a model file belongs to: a ranked guess the user confirms on the model card.

The evidence, strongest first:

* **Tensor names** (from the header only): a single-file SD 1.x, SD 2.x or SDXL checkpoint,
  SD3's joint blocks, FLUX.1's double blocks (with or without a guidance input), Chroma's
  distilled guidance layer. These say what the weights are, whatever the file is called.
* **The Hub repo** it came from, or a ``base_model`` its card names.
* **The Diffusers pipeline class** (``model_index.json``) of the repo or folder.
* **The file's name** (and its folder's): the families' ``names`` patterns.
* **GGUF metadata**: ``general.architecture``, which only narrows it down to an
  architecture (several families share one).

Each piece adds to a family's score (1 - the product of 1 - each piece's weight), and the
reasons are kept so the card can say why it guessed.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from imaginer.image_models import recipes
from imaginer.loras import files

# Diffusers pipeline classes (model_index.json's _class_name) -> families.
PIPELINES: Mapping[str, tuple[str, ...]] = {
    "StableDiffusionPipeline": ("sd1-checkpoint", "sd2-512-checkpoint", "sd2-768-checkpoint"),
    "StableDiffusionXLPipeline": ("sdxl-checkpoint", "sdxl-turbo"),
    "StableDiffusion3Pipeline": ("sd35-large", "sd35-medium", "sd3-medium"),
    "FluxPipeline": ("flux1-dev", "flux1-schnell"),
    "Flux2Pipeline": ("flux2-dev",),
    "Flux2KleinPipeline": (
        "flux2-klein-4b", "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b",
    ),
    "ChromaPipeline": ("chroma",),
    "QwenImagePipeline": ("qwen-image",),
    "QwenImage21Pipeline": ("qwen-image-21", "qwen-image-21-turbo"),
    "ZImagePipeline": ("z-image-turbo", "z-image"),
    "PixArtSigmaPipeline": ("pixart-sigma",),
    "PixArtAlphaPipeline": ("pixart-alpha-1024", "pixart-alpha-512"),
    "LongCatImagePipeline": ("longcat-image",),
    "OvisImagePipeline": ("ovis-image",),
}  # fmt: skip
_JOINT = re.compile(r"joint_blocks\.(\d+)\.")
# Folder names that say what a file is, not which model.
_GENERIC = frozenset(
    {
        "diffusion_models", "checkpoints", "stable-diffusion", "text_encoders", "clip", "vae",
        "unet", "transformer", "split_files", "models", "snapshots", "loras", "tokenizer",
        "text_encoder", "text_encoder_2", "text_encoder_3", "vae_approx", "gguf",
    }
)  # fmt: skip


@dataclass(slots=True)
class Guess:
    family: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, weight: float, reason: str) -> None:
        self.score = 1 - (1 - self.score) * (1 - weight)
        if reason not in self.reasons:
            self.reasons.append(reason)


@dataclass(frozen=True, slots=True)
class Evidence:
    """What is known about a model file: any of these may be missing."""

    path: Path | None = None  # the file (or index, or folder) on disk
    name: str | None = None  # its file name, when there is no path yet (a Hub file)
    repo: str | None = None  # the Hub repo it is in
    base_models: tuple[str, ...] = ()  # the repo card's base_model
    pipeline: str | None = None  # a Diffusers pipeline class


def detect(evidence: Evidence, catalog: recipes.Catalog | None = None) -> list[Guess]:
    """Families that fit, best first (each with a score from 0 to 1 and why)."""
    cat = catalog or recipes.catalog()
    families = cat.families(hidden=True)
    guesses: dict[str, Guess] = {}

    def add(family_id: str, weight: float, reason: str) -> None:
        family = cat.get(family_id)
        if family is not None:
            guesses.setdefault(family_id, Guess(family_id)).add(weight, reason)

    path = evidence.path
    if path is not None:
        names = files.tensor_names(path)
        if names:
            for family_id, weight, reason in _from_tensors(names):
                add(family_id, weight, reason)
        if path.suffix.lower() == ".gguf":
            arch = (files.gguf_metadata(path) or {}).get("general.architecture")
            if arch:
                for family in families:
                    if arch in family.gguf:
                        add(family.id, 0.35, f"GGUF architecture {arch}")
    for text, weight in _texts(evidence):
        for family in families:
            best = max((len(m.group(0)) for p in family.names if (m := p.search(text))), default=0)
            if best:
                # A longer match is more specific: "qwen_image_2.1_turbo" over "qwen_image".
                add(family.id, min(0.5 + best / 100, 0.75) * weight, f"its name ({text})")
    for repo in _repos(evidence):
        for family in families:
            for known in family.repos:
                if repo.casefold() == known.casefold():
                    add(family.id, 0.8, f"Hugging Face repo {known}")
                elif known.endswith("/") and repo.casefold().startswith(known.casefold()):
                    add(family.id, 0.3, f"Hugging Face owner {known}")
    if evidence.pipeline is not None and evidence.pipeline in PIPELINES:
        for family_id in PIPELINES[evidence.pipeline]:
            add(family_id, 0.6, f"Diffusers {evidence.pipeline}")
    ranked = sorted(guesses.values(), key=lambda g: (-g.score, g.family))
    return [g for g in ranked if not _hidden(cat, g.family)] + [
        g for g in ranked if _hidden(cat, g.family)
    ]


def _hidden(cat: recipes.Catalog, family_id: str) -> bool:
    family = cat.get(family_id)
    return family is None or family.hidden


def _texts(evidence: Evidence) -> Iterable[tuple[str, float]]:
    """Names to match the families' patterns against, with how much each counts: the
    file's own name most, its folder's and the repo's less."""
    if evidence.path is not None:
        yield evidence.path.name, 1.0
        folders = [p.name for p in evidence.path.parents[:4] if p.name]
        named = (f for f in folders if f.casefold() not in _GENERIC)
        yield from zip(named, (0.85, 0.7, 0.55), strict=False)
    elif evidence.name:
        yield Path(evidence.name).name, 1.0
    if evidence.repo:
        yield evidence.repo.rsplit("/", 1)[-1], 0.8
    for base in evidence.base_models:
        yield base.rsplit("/", 1)[-1], 0.8


def _repos(evidence: Evidence) -> Iterable[str]:
    if evidence.repo:
        yield evidence.repo
    yield from evidence.base_models


def _from_tensors(names: list[str]) -> Iterable[tuple[str, float, str]]:
    """Families a model's tensor names point at."""
    has = set()
    for name in names:
        for mark in _MARKS:
            if mark in name:
                has.add(mark)
    sd_unet = "input_blocks." in has
    if "conditioner.embedders.1." in has:
        yield "sdxl-checkpoint", 0.9, "SDXL checkpoint tensors"
        yield "sdxl-turbo", 0.5, "SDXL checkpoint tensors"
    elif "cond_stage_model.model.transformer." in has and sd_unet:
        for family_id in ("sd2-768-checkpoint", "sd2-512-checkpoint", "sd-turbo"):
            yield family_id, 0.6, "SD 2.x checkpoint tensors"
    elif "cond_stage_model.transformer." in has and sd_unet:
        yield "sd1-checkpoint", 0.9, "SD 1.x checkpoint tensors"
    if "joint_blocks." in has:
        depth = 1 + max((int(m.group(1)) for n in names if (m := _JOINT.search(n))), default=0)
        bundled = "text_encoders." in has
        if depth >= 38:
            yield "sd35-large", 0.85, f"SD3 tensors ({depth} joint blocks)"
        else:
            yield "sd35-medium", 0.5, f"SD3 tensors ({depth} joint blocks)"
            yield "sd3-medium", 0.75 if bundled else 0.5, f"SD3 tensors ({depth} joint blocks)"
    if "double_blocks." in has and "double_stream_modulation_img." not in has:
        if "distilled_guidance_layer." in has:
            if "nerf_" in has:
                yield "chroma-radiance", 0.9, "Chroma Radiance tensors"
            else:
                yield "chroma", 0.9, "Chroma tensors"
        elif "guidance_in." in has:
            yield "flux1-dev", 0.85, "FLUX.1 tensors with a guidance input"
        else:
            yield "flux1-schnell", 0.7, "FLUX.1 tensors without a guidance input"


_MARKS = (
    "conditioner.embedders.1.",
    "cond_stage_model.model.transformer.",
    "cond_stage_model.transformer.",
    "input_blocks.",
    "joint_blocks.",
    "text_encoders.",
    "double_blocks.",
    "double_stream_modulation_img.",
    "distilled_guidance_layer.",
    "guidance_in.",
    "nerf_",
)


def best(evidence: Evidence, catalog: recipes.Catalog | None = None) -> Guess | None:
    """The best guess, if any fits well enough to pick it for the user."""
    found = detect(evidence, catalog)
    return found[0] if found and found[0].score >= 0.5 else None
