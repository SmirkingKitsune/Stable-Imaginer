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
        # Edit models, beside the text-to-image ones they share a repo or an architecture with.
        ("flux1-kontext-dev-Q8_0.gguf", "QuantStack/FLUX.1-Kontext-dev-GGUF", "flux1-kontext"),
        ("Qwen_Image_Edit-Q8_0.gguf", "QuantStack/Qwen-Image-Edit-GGUF", "qwen-image-edit"),
        ("qwen_image_edit_2509_fp8mixed.safetensors", None, "qwen-image-edit-2509"),
        (
            "qwen-image-edit-2511-Q4_K_M.gguf",
            "unsloth/Qwen-Image-Edit-2511-GGUF",
            "qwen-image-edit-2511",
        ),
        (
            "qwen_image_edit_2511_bf16.safetensors",
            "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "qwen-image-edit-2511",
        ),
        ("longcat_image_edit_bf16.safetensors", "Comfy-Org/LongCat-Image", "longcat-image-edit"),
        ("longcat_image_bf16.safetensors", "Comfy-Org/LongCat-Image", "longcat-image"),
        ("boogu_image_edit_int8_convrot.safetensors", "Comfy-Org/Boogu-Image", "boogu-image-edit"),
        ("boogu_image_base_bf16.safetensors", "Comfy-Org/Boogu-Image", "boogu-image"),
        ("mage_flow_edit_turbo_bf16.safetensors", "Comfy-Org/Mage-Flow", "mage-flow-edit"),
        ("mage_flow_turbo_bf16.safetensors", "Comfy-Org/Mage-Flow", "mage-flow-turbo"),
        ("fibo2_edit_turbo_bf16.safetensors", "Comfy-Org/Fibo-2.0", "fibo2-edit-turbo"),
        ("fibo2_dev_int8_convrot.safetensors", "Comfy-Org/Fibo-2.0", "fibo2-dev"),
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


def test_kontext_has_flux_dev_tensors_and_its_own_name(tmp_path: Path) -> None:
    kontext = headers.safetensors(tmp_path / "flux1-kontext-dev.safetensors", headers.FLUX_DEV)
    assert _top(Evidence(path=kontext)) == "flux1-kontext"
    anonymous = headers.safetensors(tmp_path / "model.safetensors", headers.FLUX_DEV)
    assert _top(Evidence(path=anonymous)) == "flux1-dev"  # the same tensors: dev wins a tie


@pytest.mark.parametrize(
    ("pipeline", "families"),
    [
        ("FluxKontextPipeline", {"flux1-kontext"}),
        ("QwenImageEditPipeline", {"qwen-image-edit"}),
        ("QwenImageEditPlusPipeline", {"qwen-image-edit-2509", "qwen-image-edit-2511"}),
        ("LongCatImageEditPipeline", {"longcat-image-edit"}),
    ],
)
def test_edit_pipelines(pipeline: str, families: set[str]) -> None:
    guesses = detect(Evidence(pipeline=pipeline))
    assert {g.family for g in guesses} == families


def test_the_repo_tells_qwen_edit_2509_from_2511() -> None:
    plus = "QwenImageEditPlusPipeline"
    assert _top(Evidence(pipeline=plus, repo="Qwen/Qwen-Image-Edit-2511")) == "qwen-image-edit-2511"
    assert _top(Evidence(pipeline=plus, repo="Qwen/Qwen-Image-Edit-2509")) == "qwen-image-edit-2509"
