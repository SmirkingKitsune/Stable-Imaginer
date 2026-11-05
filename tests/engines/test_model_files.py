"""What each engine can load: llama-server reads GGUF only, from a split GGUF's first shard;
sd-server also reads sharded safetensors through their index (stable-diffusion.cpp #1769),
so a single shard on its own, which would load part of a model, is caught here."""

import json
import struct
from pathlib import Path

import pytest

from imaginer.core.config import SdCppLaunch
from imaginer.engines import launch
from tests.conftest import sd_config


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * 60)
    return path


def _safetensors(path: Path, payload: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = b'{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload)
    return path


def _sharded(folder: Path, base: str = "diffusion_pytorch_model", total: int = 2) -> Path:
    names = [f"{base}-{n:05d}-of-{total:05d}.safetensors" for n in range(1, total + 1)]
    for name in names:
        _safetensors(folder / name, payload=100)
    index = folder / f"{base}.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {f"t{i}": n for i, n in enumerate(names)}}))
    return index


def _check(path: Path, engine: str) -> launch.FileCheck:
    return launch.check_file("diffusion_model", "--diffusion-model", "model", path, engine=engine)


def test_sd_server_loads_sharded_safetensors_through_their_index(tmp_path: Path) -> None:
    index = _sharded(tmp_path / "transformer")
    shards = sum(p.stat().st_size for p in index.parent.glob("*.safetensors"))
    check = _check(index, "sdcpp")
    assert (check.ok, check.format, check.size_bytes) == (True, "safetensors", shards)
    # One shard alone would load half the model: point at the index instead.
    shard = _check(
        tmp_path / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors", "sdcpp"
    )
    assert shard.problem == (
        "one of 2 shards: point at diffusion_pytorch_model.safetensors.index.json to load them all"
    )


def test_a_transformers_folder_loads_through_its_root_index(tmp_path: Path) -> None:
    _sharded(tmp_path / "text_encoder", base="model", total=3)
    assert _check(tmp_path / "text_encoder", "sdcpp").ok
    # A Diffusers pipeline's own folder isn't one model.
    (tmp_path / "model_index.json").write_text("{}")
    pipeline = _check(tmp_path, "sdcpp")
    assert pipeline.problem is not None and pipeline.problem.startswith(
        "a Diffusers pipeline folder"
    )


def test_an_index_with_missing_shards_says_so(tmp_path: Path) -> None:
    index = _sharded(tmp_path)
    (tmp_path / "diffusion_pytorch_model-00002-of-00002.safetensors").unlink()
    check = _check(index, "sdcpp")
    assert check.problem == (
        "1 of its 2 shards are missing, e.g. diffusion_pytorch_model-00002-of-00002.safetensors"
    )
    (tmp_path / "broken.json").write_text("{}")
    assert (
        _check(tmp_path / "broken.json", "sdcpp").problem
        == "not a safetensors index (no weight_map)"
    )


def test_llama_server_loads_gguf_only_from_the_first_shard(tmp_path: Path) -> None:
    def model(path: Path) -> launch.FileCheck:
        return launch.check_file("model", "-m", "model", path, engine="llamacpp")

    assert model(_gguf(tmp_path / "m-00001-of-00003.gguf")).ok
    second = model(_gguf(tmp_path / "m-00002-of-00003.gguf"))
    assert second.problem == "shard 2 of 3: point at the first one, m-00001-of-00003.gguf"
    assert model(_safetensors(tmp_path / "m.safetensors")).problem == launch.GGUF_ONLY
    assert model(_sharded(tmp_path / "hf")).problem == launch.GGUF_ONLY


@pytest.mark.parametrize("engine", ["sdcpp", "llamacpp", None])
def test_folders_are_not_model_files(tmp_path: Path, engine: str | None) -> None:
    check = launch.check_file("vae", "--vae", "VAE", tmp_path, engine=engine)
    assert (check.exists, check.problem) == (True, "a folder, not a model file")


def test_sd_server_is_started_with_the_index(tmp_path: Path) -> None:
    index = _sharded(tmp_path / "transformer")
    config = sd_config("http://127.0.0.1:1234", launch=SdCppLaunch(diffusion_model=index))
    argv = launch.command(config, Path("sd-server"))
    assert argv[1:3] == ["--diffusion-model", str(index)]
    shard = tmp_path / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors"
    config = sd_config("http://127.0.0.1:1234", launch=SdCppLaunch(diffusion_model=shard))
    with pytest.raises(launch.LaunchError, match="one of 2 shards"):
        launch.command(config, Path("sd-server"))
