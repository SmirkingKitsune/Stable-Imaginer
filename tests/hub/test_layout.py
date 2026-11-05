"""Repo layouts, sorted into what the engines load. The listings are real ones, captured from
the Hub, so these follow how the big publishers actually lay their repos out."""

import pytest

from imaginer.hub.layout import (
    RepoFile,
    RepoLayout,
    classify,
    quant_of,
    roles_for,
    shard_base,
    suggest,
)
from tests.hub.conftest import listing


def _layout(repo_id: str) -> RepoLayout:
    files, meta = listing(repo_id)
    return classify(files, repo_id=repo_id, **meta)


def _picks(engine: str, repo_id: str, **quants: str) -> dict[str, str | None]:
    roles = roles_for(engine, _layout(repo_id))
    return {
        key: (best.id if (best := suggest(found, key=key, quant=quants.get(key))) else None)
        for key, found in roles.items()
    }


def test_a_flat_gguf_repo_offers_its_quants_and_projectors() -> None:
    layout = _layout("unsloth/Qwen3-VL-8B-Instruct-GGUF")
    assert layout.kind == "gguf"
    assert layout.task == "llm"
    models = [v for v in layout.variants if v.component == "model"]
    projectors = [v for v in layout.variants if v.component == "mmproj"]
    assert len(models) == 26 and len(projectors) == 3
    assert {v.quant for v in models} >= {"Q4_K_M", "UD-Q4_K_XL", "IQ4_XS", "BF16", "UD-IQ1_S"}
    assert not any("imatrix" in v.id for v in layout.variants)  # calibration data, not a model
    assert _picks("llamacpp", "unsloth/Qwen3-VL-8B-Instruct-GGUF") == {
        "model": "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-F16.gguf",
    }


def test_a_quant_asked_for_wins_over_the_default() -> None:
    picks = _picks("llamacpp", "bartowski/Qwen_Qwen3-VL-8B-Instruct-GGUF", model="q6_k_l")
    assert picks["model"] == "Qwen_Qwen3-VL-8B-Instruct-Q6_K_L.gguf"
    # bartowski names precisions in lower case; they still sort as the same quants.
    assert picks["mmproj"] == "mmproj-Qwen_Qwen3-VL-8B-Instruct-f16.gguf"


def test_big_models_split_into_quant_folders_load_from_their_first_shard() -> None:
    layout = _layout("unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF")
    q4 = next(v for v in layout.variants if v.quant == "Q4_K_M")
    assert q4.id == "Q4_K_M/Qwen3-VL-235B-A22B-Instruct-Q4_K_M-00001-of-00003.gguf"
    assert [f.path.rsplit("-", 3)[-3] for f in q4.files] == ["00001", "00002", "00003"]
    assert (q4.sharded, q4.indexed, q4.size) == (True, False, 142154076192)
    assert _picks("llamacpp", "unsloth/Qwen3-VL-235B-A22B-Instruct-GGUF")["model"] == q4.id
    # sd-server can't open a split GGUF: only the single-file quant fits there.
    sd = roles_for("sdcpp", layout)
    assert all(not v.sharded for found in sd.values() for v in found)


def test_an_image_gguf_repo_fills_the_diffusion_model() -> None:
    assert _picks("sdcpp", "leejet/Qwen-Image-2.1-GGUF") == {
        "diffusion_model": "qwen_image_2.1-Q4_K.gguf",
        "vae": None,
        "llm": None,
        "llm_vision": None,
    }
    assert _picks("llamacpp", "leejet/Qwen-Image-2.1-GGUF") == {"model": None, "mmproj": None}


def test_an_llm_gguf_repo_is_also_sd_servers_text_encoder() -> None:
    assert _picks("sdcpp", "Qwen/Qwen3-VL-8B-Instruct-GGUF") == {
        "diffusion_model": None,
        "vae": None,
        "llm": "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        "llm_vision": "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    }


def test_comfy_repackages_map_folders_to_settings() -> None:
    layout = _layout("Comfy-Org/Qwen-Image-2.1")
    assert layout.kind == "comfy"
    by_component = {v.component for v in layout.variants}
    assert by_component == {"diffusion_model", "vae", "text_encoder", "lora", "other"}
    assert _picks("sdcpp", "Comfy-Org/Qwen-Image-2.1") == {
        "diffusion_model": "diffusion_models/qwen_image_2.1_bf16.safetensors",
        "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
        "llm": "text_encoders/qwen3vl_8b_bf16.safetensors",
        "llm_vision": None,
    }
    # ComfyUI's split_files/ folder, and the official files over non_official/ ones.
    picks = _picks("sdcpp", "Comfy-Org/Qwen-Image_ComfyUI")
    assert picks["diffusion_model"] == "split_files/diffusion_models/qwen_image_bf16.safetensors"
    assert picks["vae"] == "split_files/vae/qwen_image_vae.safetensors"


def test_a_diffusers_pipeline_is_whole_for_diffusers_and_in_parts_for_sd_server() -> None:
    layout = _layout("Qwen/Qwen-Image-2.1")
    assert layout.kind == "diffusers"
    pipeline = layout.pipeline
    assert pipeline is not None
    assert pipeline.pipeline_class == "QwenImage21Pipeline"
    assert pipeline.components == ("processor", "scheduler", "text_encoder", "transformer", "vae")
    paths = {f.path for f in pipeline.files}
    assert "model_index.json" in paths and "assets/qr.png" not in paths
    assert "transformer/diffusion_pytorch_model.safetensors.index.json" in paths
    # sd-server reads sharded safetensors through their index (stable-diffusion.cpp #1769).
    assert _picks("sdcpp", "Qwen/Qwen-Image-2.1") == {
        "diffusion_model": "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "llm": "text_encoder/model.safetensors.index.json",
        "llm_vision": None,
    }
    transformer = layout.variant("transformer/diffusion_pytorch_model.safetensors.index.json")
    assert transformer is not None
    assert (transformer.name, transformer.indexed, len(transformer.files)) == (
        "transformer",
        True,
        3,
    )


def test_a_transformers_checkpoint_points_at_its_quantized_versions() -> None:
    layout = _layout("Qwen/Qwen3-VL-8B-Instruct")
    assert layout.kind == "transformers"
    assert any("quantized versions" in note for note in layout.notes)
    assert _picks("llamacpp", "Qwen/Qwen3-VL-8B-Instruct") == {"model": None, "mmproj": None}
    assert _picks("sdcpp", "Qwen/Qwen3-VL-8B-Instruct")["llm"] == "model.safetensors.index.json"


def test_loose_safetensors_are_sorted_by_name_and_repo() -> None:
    picks = _picks("sdcpp", "unsloth/Qwen-Image-2.1-FP8")
    assert picks["vae"] == "vae/qwen_image_2.1_vae_bf16.safetensors"
    assert picks["llm"] == "Qwen-Image-2.1-text_encoder-FP8.safetensors"
    assert picks["diffusion_model"] == "Qwen-Image-2.1-FP8.safetensors"
    # A Diffusers component published on its own: the repo's name says it's a VAE.
    vaes = roles_for("sdcpp", _layout("madebyollin/texture-fix-vae-for-qwen-image-2.1"))["vae"]
    assert {v.id for v in vaes} == {
        "diffusion_pytorch_model.safetensors",
        "texture_fix_vae_for_qwen_image_2.1_bf16.safetensors",
    }


def test_incomplete_shards_are_skipped_with_a_note() -> None:
    layout = classify(
        [RepoFile("m-00001-of-00003.gguf", 10), RepoFile("m-00003-of-00003.gguf", 10)]
    )
    assert layout.variants == ()
    assert any("only 2 of its 3 shards" in note for note in layout.notes)


def test_pickled_weights_are_named_as_unusable() -> None:
    layout = classify([RepoFile("model.ckpt", 10), RepoFile("README.md", 1)])
    assert (layout.kind, layout.variants) == ("unknown", ())
    assert any("pickled" in note for note in layout.notes)


@pytest.mark.parametrize(
    ("name", "quant"),
    [
        ("Qwen3-VL-8B-Instruct-UD-Q4_K_XL", "UD-Q4_K_XL"),
        ("qwen_image_2.1-Q4_K", "Q4_K"),
        ("Qwen_Qwen3-VL-8B-Instruct-bf16", "BF16"),
        ("mmproj-F16", "F16"),
        ("qwen_image_fp8_e4m3fn", "FP8_E4M3FN"),
        ("Qwen-Image-2.1-INT8-ConvRot", "INT8_CONVROT"),
        ("Qwen3-VL-8B-Instruct", None),
    ],
)
def test_quants_are_read_from_names(name: str, quant: str | None) -> None:
    assert quant_of(name) == quant


def test_shard_suffixes_come_off() -> None:
    assert shard_base("Q4_K_M/x-Q4_K_M-00002-of-00003.gguf") == "x-Q4_K_M"
    assert shard_base("diffusion_pytorch_model.safetensors") == "diffusion_pytorch_model"
