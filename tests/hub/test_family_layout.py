"""Matching real repos' files (captured listings) to the sd-server slots of families beyond
Qwen-Image-2.1: checkpoints, CLIP-L/CLIP-G/T5 text encoders, FLUX's and Z-Image's parts."""

import pytest

from imaginer.hub.layout import RepoFile, RepoLayout, classify, roles_for, suggest
from tests.hub.conftest import listing


def _layout(repo_id: str) -> RepoLayout:
    files, extra = listing(repo_id)
    return classify(
        files,
        repo_id=repo_id,
        pipeline_tag=extra["pipeline_tag"],
        model_index=extra["model_index"],
    )


def _pick(repo_id: str, *slots: str) -> dict[str, str | None]:
    roles = roles_for("sdcpp", _layout(repo_id))
    return {slot: (best.id if (best := suggest(roles[slot], key=slot)) else None) for slot in slots}


@pytest.mark.parametrize(
    ("repo", "picks"),
    [
        (
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            {"model": "v1-5-pruned-emaonly.safetensors"},
        ),
        (
            "stabilityai/stable-diffusion-xl-base-1.0",
            {"model": "sd_xl_base_1.0.safetensors"},  # not a Diffusers part, nor the LoRA
        ),
        (
            "black-forest-labs/FLUX.1-schnell",
            {
                "diffusion_model": "flux1-schnell.safetensors",
                "vae": "ae.safetensors",  # over vae/diffusion_pytorch_model.safetensors
                "clip_l": "text_encoder/model.safetensors",  # CLIPTextModel
                "t5xxl": "text_encoder_2/model.safetensors.index.json",  # T5EncoderModel
                "llm": None,
            },
        ),
        (
            "comfyanonymous/flux_text_encoders",
            {"clip_l": "clip_l.safetensors", "t5xxl": "t5xxl_fp16.safetensors", "llm": None},
        ),
        (
            "Comfy-Org/stable-diffusion-3.5-fp8",
            {
                "model": "sd3.5_large_fp8_scaled.safetensors",
                "clip_l": "text_encoders/clip_l.safetensors",
                "clip_g": "text_encoders/clip_g.safetensors",
                "t5xxl": "text_encoders/t5xxl_fp16.safetensors",
                "llm": None,  # a checkpoint in an untagged repo is no LLM
            },
        ),
        (
            "Comfy-Org/z_image_turbo",
            {
                "diffusion_model": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
                "vae": "split_files/vae/ae.safetensors",
                "t5xxl": None,
            },
        ),
        (
            "Comfy-Org/flux2-klein-4B",
            {
                "vae": "split_files/vae/flux2-vae.safetensors",
                "diffusion_model": "split_files/diffusion_models/flux-2-klein-4b.safetensors",
            },
        ),
        ("leejet/FLUX.2-klein-4B-GGUF", {"diffusion_model": "flux-2-klein-4b-Q4_0.gguf"}),
        ("lodestones/Chroma", {"vae": "ae.safetensors"}),
    ],
)
def test_each_slot_gets_the_right_file(repo: str, picks: dict[str, str | None]) -> None:
    assert _pick(repo, *picks) == picks


def test_text_encoders_are_told_apart() -> None:
    layout = _layout("Comfy-Org/stable-diffusion-3.5-fp8")
    kinds = {v.id: v.encoder for v in layout.variants if v.component == "text_encoder"}
    assert kinds["text_encoders/clip_l.safetensors"] == "clip_l"
    assert kinds["text_encoders/clip_g.safetensors"] == "clip_g"
    assert kinds["text_encoders/t5xxl_fp8_e4m3fn.safetensors"] == "t5xxl"
    z_image = _layout("Comfy-Org/z_image_turbo")
    assert {v.encoder for v in z_image.variants if v.component == "text_encoder"} == {"llm"}
    sdxl = _layout("stabilityai/stable-diffusion-xl-base-1.0")
    second = next(v for v in sdxl.variants if v.id == "text_encoder_2/model.safetensors")
    assert second.encoder == "clip_g"  # SDXL's CLIPTextModelWithProjection is OpenCLIP bigG


def test_tokenizers_are_files_to_load_too() -> None:
    layout = _layout("black-forest-labs/FLUX.1-schnell")
    tokenizers = [v for v in layout.variants if v.component == "tokenizer"]
    assert [v.id for v in tokenizers] == ["tokenizer_2/tokenizer.json"]
    assert roles_for("sdcpp", layout)["tokenizer"] == tokenizers
    assert roles_for("llamacpp", layout)["model"] == []


def test_a_quantizers_untagged_gguf_of_a_chat_model_is_a_text_encoder() -> None:
    """mradermacher's GGUF repos carry no task tag; an Instruct/VL model in one is an LLM
    text encoder (Qwen Image's and its edit models' Qwen2.5-VL), not an image model."""
    files = [
        RepoFile("Qwen2.5-VL-7B-Instruct.Q8_0.gguf", 8098524160, None),
        RepoFile("Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf", 1354162912, None),
    ]
    layout = classify(files, repo_id="mradermacher/Qwen2.5-VL-7B-Instruct-GGUF", pipeline_tag=None)
    roles = roles_for("sdcpp", layout)
    assert [v.id for v in roles["llm"]] == ["Qwen2.5-VL-7B-Instruct.Q8_0.gguf"]
    assert [v.id for v in roles["llm_vision"]] == ["Qwen2.5-VL-7B-Instruct.mmproj-f16.gguf"]
    kontext = classify(
        [RepoFile("flux1-kontext-dev-Q8_0.gguf", 12714452256, None)],
        repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF",
        pipeline_tag=None,
    )
    assert roles_for("sdcpp", kontext)["llm"] == []  # an image model's GGUF stays one
