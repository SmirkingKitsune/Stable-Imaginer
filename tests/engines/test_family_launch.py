"""sd-server's command line for each shape of model family: a checkpoint, a diffusion model
with CLIP/T5 encoders, with an LLM encoder, with a tokenizer, with no VAE, with an
unconditional model; and what each family requires before it can start."""

from pathlib import Path
from typing import Any

import pytest

from imaginer.core.config import SdCppLaunch
from imaginer.engines import launch
from tests.conftest import sd_config
from tests.image_models import headers


def _files(tmp_path: Path, **slots: str) -> dict[str, Path]:
    out = {}
    for slot, name in slots.items():
        path = tmp_path / name
        if name.endswith(".json"):
            path.write_text('{"model": {"type": "BPE"}}', encoding="utf-8")
        elif name.endswith(".gguf"):
            headers.gguf(path, ["x"])
        else:
            headers.safetensors(path, ["x"])
        out[slot] = path
    return out


def _argv(tmp_path: Path, family: str | None, **launch_fields: Any) -> list[str]:
    config = sd_config("http://127.0.0.1:1234", family=family, launch=SdCppLaunch(**launch_fields))
    argv = launch.command(config, Path("sd-server"))
    return [a.replace(str(tmp_path), "~") for a in argv[1:]]


def test_a_checkpoint_family_loads_one_file_with_model(tmp_path: Path) -> None:
    files = _files(tmp_path, model="sd_xl_base_1.0.safetensors", vae="sdxl_vae.safetensors")
    assert _argv(tmp_path, "sdxl-checkpoint", **files)[:4] == [
        "--model", str(Path("~/sd_xl_base_1.0.safetensors")),
        "--vae", str(Path("~/sdxl_vae.safetensors")),
    ]  # fmt: skip


def test_flux_loads_a_diffusion_model_with_clip_and_t5(tmp_path: Path) -> None:
    files = _files(
        tmp_path,
        diffusion_model="flux1-dev-Q8_0.gguf",
        vae="ae.safetensors",
        clip_l="clip_l.safetensors",
        t5xxl="t5xxl_fp16.safetensors",
    )
    flags = [a for a in _argv(tmp_path, "flux1-dev", **files) if a.startswith("--")]
    assert flags == [
        "--diffusion-model",
        "--vae",
        "--clip_l",
        "--t5xxl",
        "--listen-ip",
        "--listen-port",
    ]


def test_a_tokenizer_and_a_pixel_space_family(tmp_path: Path) -> None:
    moss = _files(
        tmp_path,
        diffusion_model="moss.safetensors",
        llm="qwen3.5.gguf",
        vae="flux2_ae.safetensors",
        tokenizer="tokenizer.json",
    )
    argv = _argv(tmp_path, "moss-image-base", **moss)
    assert argv[argv.index("--tokenizer") + 1] == str(Path("~/tokenizer.json"))
    radiance = _files(tmp_path, diffusion_model="radiance.gguf", t5xxl="t5xxl.safetensors")
    flags = [a for a in _argv(tmp_path, "chroma-radiance", **radiance) if a.startswith("--")]
    assert "--vae" not in flags and flags[:2] == ["--diffusion-model", "--t5xxl"]


def test_ideogram_has_an_unconditional_model(tmp_path: Path) -> None:
    files = _files(
        tmp_path,
        diffusion_model="ideogram4.gguf",
        uncond_diffusion_model="ideogram4_uncond.gguf",
        llm="qwen3vl.gguf",
        vae="flux2_ae.safetensors",
    )
    flags = [a for a in _argv(tmp_path, "ideogram4", **files) if a.startswith("--")]
    assert flags[:2] == ["--diffusion-model", "--uncond-diffusion-model"]


@pytest.mark.parametrize(
    ("family", "slots", "missing"),
    [
        ("flux1-dev", {"diffusion_model": "f.gguf"}, "--clip_l, --t5xxl, --vae or --tae not set"),
        ("sdxl-checkpoint", {"vae": "v.safetensors"}, "--model not set"),
        ("lens", {"diffusion_model": "l.gguf", "llm": "g.gguf", "vae": "v.safetensors"},
         "--tokenizer not set"),
        (None, {"vae": "v.safetensors"}, "--model or --diffusion-model not set"),
    ],
)  # fmt: skip
def test_a_family_says_what_is_missing(
    tmp_path: Path, family: str | None, slots: dict[str, str], missing: str
) -> None:
    with pytest.raises(launch.LaunchError, match=missing):
        _argv(tmp_path, family, **_files(tmp_path, **slots))


def test_the_familys_model_args_and_vae_format(tmp_path: Path) -> None:
    files = _files(
        tmp_path, diffusion_model="chroma.gguf", t5xxl="t5.safetensors", vae="ae.sft.safetensors"
    )
    argv = _argv(tmp_path, "chroma", **files)
    assert argv[argv.index("--model-args") + 1] == "chroma_use_dit_mask=false"
    # The launch block's own win, and add to the family's.
    argv = _argv(
        tmp_path,
        "chroma",
        **files,
        model_args={"chroma_use_dit_mask": True, "x": 1},
        vae_format="flux",
    )
    assert argv[argv.index("--model-args") + 1] == "chroma_use_dit_mask=true,x=1"
    assert argv[argv.index("--vae-format") + 1] == "flux"


def test_an_engine_lists_its_familys_slots(tmp_path: Path) -> None:
    config = sd_config("http://127.0.0.1:1234", family="sd35-large", launch=SdCppLaunch())
    assert [c.key for c in launch.check_files(config)] == [
        "model", "vae", "tae", "clip_l", "clip_g", "t5xxl"
    ]  # fmt: skip
    legacy = sd_config("http://127.0.0.1:1234", launch=SdCppLaunch(tokenizer=tmp_path / "t.json"))
    assert [c.key for c in launch.check_files(legacy)] == [
        "diffusion_model", "vae", "llm", "llm_vision", "tokenizer"
    ]  # fmt: skip


def test_a_tokenizer_must_be_a_tokenizer_json(tmp_path: Path) -> None:
    good = _files(tmp_path, tokenizer="tokenizer.json")["tokenizer"]
    assert launch.check_file("tokenizer", "--tokenizer", "tokenizer", good).ok
    assert launch.check_file("tokenizer", "--tokenizer", "tokenizer", tmp_path).problem == (
        "a folder: point at its tokenizer.json"
    )
    weights = _files(tmp_path, x="w.safetensors")["x"]
    assert launch.check_file("tokenizer", "--tokenizer", "tokenizer", weights).problem == (
        "not a tokenizer.json"
    )
