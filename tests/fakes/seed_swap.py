"""Stub model files and two model cards for checking an edit card with fakes
(config/dev-swap.yaml): Qwen-Image-2.1 as the engine's model card, Qwen-Image-Edit-2511 as
its edit card. The files are headers only, enough for the library and the fake sd-server.
Also a FLUX.1 Kontext stub, to import by hand and see it detected.

    uv run python -m tests.fakes.seed_swap [--url http://127.0.0.1:8702]
"""

import argparse
from pathlib import Path
from typing import Any

import httpx

from tests.image_models import headers


def _stubs(folder: Path) -> dict[str, Path]:
    return {
        "main": headers.safetensors(folder / "qwen_image_2.1_bf16.safetensors", ["img_in.weight"]),
        "main_llm": headers.gguf(folder / "Qwen3VL-8B-Instruct-Q4_K_M.gguf", ["x"]),
        "main_vision": headers.gguf(folder / "mmproj-Qwen3VL-8B-Instruct-F16.gguf", ["x"]),
        "main_vae": headers.safetensors(folder / "qwen_image_2.1_vae_bf16.safetensors", ["x"]),
        "edit": headers.gguf(folder / "qwen-image-edit-2511-Q4_K_M.gguf", ["x"], arch="qwen_image"),
        "edit_llm": headers.safetensors(folder / "qwen_2.5_vl_7b.safetensors", ["x"]),
        "edit_vae": headers.safetensors(folder / "qwen_image_vae.safetensors", ["x"]),
        "kontext": headers.gguf(folder / "flux1-kontext-dev-Q8_0.gguf", ["x"], arch="flux"),
    }


def _card(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/v1/models", json=body)
    resp.raise_for_status()
    card: dict[str, Any] = resp.json()
    print(f"card {card['id']}: {card['name']} ({card['family']}), ready={card['ready']}")  # noqa: T201
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8702")
    parser.add_argument("--dir", type=Path, default=Path(".data/dev-swap/stubs"))
    args = parser.parse_args()
    files = {k: str(v.resolve()) for k, v in _stubs(args.dir).items()}
    with httpx.Client(base_url=args.url, timeout=60) as client:
        main_card = _card(
            client,
            {
                "path": files["main"],
                "components": {
                    "llm": files["main_llm"],
                    "llm_vision": files["main_vision"],
                    "vae": files["main_vae"],
                },
            },
        )
        edit_card = _card(
            client,
            {
                "path": files["edit"],
                "components": {"llm": files["edit_llm"], "vae": files["edit_vae"]},
            },
        )
        client.post(
            f"/v1/models/{main_card['id']}/use", json={"engine": "sdcpp", "start": True}
        ).raise_for_status()
        client.post(
            f"/v1/models/{edit_card['id']}/use", json={"engine": "sdcpp", "role": "edit"}
        ).raise_for_status()
    print(f"sdcpp runs {main_card['name']} and edits with {edit_card['name']}")  # noqa: T201
    print(f"Kontext stub to import: {files['kontext']}")  # noqa: T201


if __name__ == "__main__":
    main()
