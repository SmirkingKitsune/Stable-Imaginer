"""Telling a model file's family: from its tensors, GGUF metadata, name and Hub repo."""

from pathlib import Path

import pytest

from imaginer.image_models.detect import Evidence, best, detect
from imaginer.loras import files
from tests.image_models import headers


def _top(evidence: Evidence) -> str | None:
    guess = best(evidence)
    return guess.family if guess is not None else None


@pytest.mark.parametrize(
    ("names", "family"),
    [
        (headers.SD1, "sd1-checkpoint"),
        (headers.SDXL, "sdxl-checkpoint"),
        (headers.FLUX_DEV, "flux1-dev"),
        (headers.FLUX_SCHNELL, "flux1-schnell"),
        (headers.CHROMA, "chroma"),
        (headers.SD35_LARGE, "sd35-large"),
    ],
)
def test_tensor_names_tell_the_family_whatever_the_file_is_called(
    tmp_path: Path, names: list[str], family: str
) -> None:
    path = headers.safetensors(tmp_path / "my-favourite-model_v9.safetensors", names)
    assert _top(Evidence(path=path)) == family


def test_a_ggufs_architecture_and_name_together(tmp_path: Path) -> None:
    path = headers.gguf(tmp_path / "flux1-dev-Q8_0.gguf", ["x.weight"], arch="flux")
    assert files.gguf_metadata(path) == {"general.architecture": "flux"}
    guesses = detect(Evidence(path=path))
    assert guesses[0].family == "flux1-dev"
    assert any("GGUF architecture flux" in r for r in guesses[0].reasons)


@pytest.mark.parametrize(
    ("name", "repo", "family"),
    [
        ("qwen_image_2.1_bf16.safetensors", None, "qwen-image-21"),
        ("qwen_image_2.1_turbo_bf16.safetensors", None, "qwen-image-21-turbo"),
        ("qwen_image_bf16.safetensors", None, "qwen-image"),
        ("z_image_turbo_bf16.safetensors", "Comfy-Org/z_image_turbo", "z-image-turbo"),
        ("flux-2-klein-4b-Q8_0.gguf", "leejet/FLUX.2-klein-4B-GGUF", "flux2-klein-4b"),
        ("flux-2-klein-base-4b.safetensors", None, "flux2-klein-base-4b"),
        (
            "sd_xl_base_1.0.safetensors",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl-checkpoint",
        ),
        ("v1-5-pruned-emaonly.safetensors", None, "sd1-checkpoint"),
        ("Chroma1-Radiance-v0.4-Q8_0.gguf", None, "chroma-radiance"),
        ("sd3.5_large_fp8_scaled.safetensors", None, "sd35-large"),
    ],
)
def test_names_and_repos(name: str, repo: str | None, family: str) -> None:
    assert _top(Evidence(name=name, repo=repo)) == family


def test_a_hub_repo_and_pipeline_class_point_at_a_family() -> None:
    assert _top(Evidence(repo="black-forest-labs/FLUX.1-schnell")) == "flux1-schnell"
    guesses = detect(Evidence(pipeline="StableDiffusionXLPipeline"))
    assert guesses[0].family in {"sdxl-checkpoint", "sdxl-turbo"}
    assert best(Evidence(name="mystery.safetensors")) is None


def test_the_folders_a_file_is_in_count_too(tmp_path: Path) -> None:
    path = headers.safetensors(
        tmp_path / "Comfy-Org" / "Qwen-Image-2.1" / "diffusion_models" / "model.safetensors", ["x"]
    )
    assert _top(Evidence(path=path)) == "qwen-image-21"
