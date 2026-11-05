import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from imaginer.hub.layout import RepoFile
from tests.fakes.fake_hub import FakeHub, FakeRepo, gguf, weights
from tests.fakes.live import LiveServer

LISTINGS = Path(__file__).with_name("listings.json")  # real Hub listings, captured 2026-10-30


@pytest.fixture(autouse=True)
def isolated_hf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never read this machine's Hub token or cache: point HF_HOME at an empty folder."""
    home = tmp_path / "hf-home"
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH", "HF_HUB_CACHE",
                 "HUGGINGFACE_HUB_CACHE", "HF_ENDPOINT"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(home))
    return home


def listing(repo_id: str) -> tuple[list[RepoFile], dict[str, Any]]:
    """A captured repo's files, and its pipeline_tag and model_index."""
    record = json.loads(LISTINGS.read_text(encoding="utf-8"))[repo_id]
    files = [RepoFile(f[0], f[1], f[2] if len(f) > 2 else None) for f in record["files"]]
    return files, {"pipeline_tag": record["pipeline_tag"], "model_index": record.get("model_index")}


# Small stand-ins for the real repos' shapes.
LLM_REPO = "acme/Chat-8B-GGUF"
IMAGE_REPO = "acme/Image-2-GGUF"
PIPELINE_REPO = "acme/Image-2"
GATED_REPO = "acme/Gated-GGUF"
MODEL_INDEX = {
    "_class_name": "QwenImage21Pipeline",
    "_diffusers_version": "0.42.1",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}


def _index(shards: dict[str, bytes]) -> bytes:
    return json.dumps({"weight_map": {f"w{i}": name for i, name in enumerate(shards)}}).encode()


@pytest.fixture
def hub() -> FakeHub:
    transformer = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": weights(3000, 5),
        "diffusion_pytorch_model-00002-of-00002.safetensors": weights(1000, 6),
    }
    return FakeHub(
        repos={
            LLM_REPO: FakeRepo(
                {
                    "README.md": b"# chat",
                    "Chat-8B-Q4_K_M.gguf": gguf(5000, 1),
                    "Chat-8B-Q8_0.gguf": gguf(8000, 2),
                    "mmproj-Chat-8B-F16.gguf": gguf(1200, 3),
                },
                pipeline_tag="image-text-to-text",
                tags=["gguf"],
                base_model="acme/Chat-8B",
            ),
            IMAGE_REPO: FakeRepo(
                {"image-2-Q4_K.gguf": gguf(4000, 4)}, pipeline_tag="text-to-image", tags=["gguf"]
            ),
            PIPELINE_REPO: FakeRepo(
                {
                    "model_index.json": json.dumps(MODEL_INDEX).encode(),
                    "scheduler/scheduler_config.json": b"{}",
                    "transformer/config.json": b"{}",
                    **{f"transformer/{k}": v for k, v in transformer.items()},
                    "transformer/diffusion_pytorch_model.safetensors.index.json": _index(
                        transformer
                    ),
                    "vae/config.json": b"{}",
                    "vae/diffusion_pytorch_model.safetensors": weights(700, 7),
                    "assets/sample.png": b"png",
                },
                pipeline_tag="text-to-image",
                tags=["diffusers", "safetensors"],
            ),
            GATED_REPO: FakeRepo({"g-Q4_K_M.gguf": gguf(900, 8)}, tags=["gguf"], gated=True),
        }
    )


@pytest.fixture
def hub_url(hub: FakeHub) -> Iterator[str]:
    with LiveServer(hub.app()) as live:
        yield live.url
