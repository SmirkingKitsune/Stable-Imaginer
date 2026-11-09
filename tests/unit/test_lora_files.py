"""Reading a LoRA's and a model's headers to guess whether one applies to the other."""

import json
import struct
from pathlib import Path

from imaginer.hub.client import _card_images
from imaginer.loras import files


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_file(path: Path, tensors: dict[str, list[int]]) -> Path:
    """A GGUF v3 header: one string key, then the tensor table (dims innermost first)."""
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), 2)
    out += _string("general.architecture") + struct.pack("<I", 8) + _string("qwen_image")
    out += _string("general.tags") + struct.pack("<IIQ", 9, 8, 2) + _string("a") + _string("bc")
    for name, shape in tensors.items():
        out += _string(name) + struct.pack("<I", len(shape))
        out += b"".join(struct.pack("<Q", d) for d in reversed(shape))
        out += struct.pack("<IQ", 0, 0)
    path.write_bytes(out)
    return path


def safetensors_file(path: Path, tensors: dict[str, list[int]], meta: dict[str, str]) -> Path:
    header: dict[str, object] = {"__metadata__": meta}
    header |= {n: {"dtype": "F16", "shape": s, "data_offsets": [0, 0]} for n, s in tensors.items()}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def test_a_gguf_models_modules_and_widths(tmp_path: Path) -> None:
    model = gguf_file(
        tmp_path / "qwen-image-2.1-Q4_K_M.gguf",
        {
            "model.diffusion_model.blocks.0.attn.to_q.weight": [64, 32],
            "model.diffusion_model.blocks.0.mlp.gate_up.weight": [128, 32],
            "model.diffusion_model.blocks.0.mlp.out.weight": [32, 64],
        },
    )
    assert files.gguf_tensors(model) == [
        ("model.diffusion_model.blocks.0.attn.to_q.weight", [64, 32]),
        ("model.diffusion_model.blocks.0.mlp.gate_up.weight", [128, 32]),
        ("model.diffusion_model.blocks.0.mlp.out.weight", [32, 64]),
    ]
    modules = files.model_file_modules(model)
    assert modules is not None
    # A fused layer counts under the names a LoRA patches its parts by.
    assert modules["blocks_0_mlp_gate_layer"] == modules["blocks_0_mlp_proj"] == 32

    lora = {
        "transformer.blocks.0.attn.to_q.lora_A.weight": [4, 32],
        "transformer.blocks.0.attn.to_q.lora_B.weight": [64, 4],
        "transformer.blocks.0.mlp.gate_layer.lora_A.weight": [4, 32],
        "transformer.blocks.0.mlp.proj.lora_A.weight": [4, 32],
    }
    facts = files.lora_facts(
        safetensors_file(tmp_path / "a.safetensors", lora, {"trigger_words": "a, b, a"})
    )
    assert (facts.format, facts.rank, facts.words) == ("PEFT keys", 4, ("a", "b"))
    assert files.matched(facts.modules, modules) == 3
    # The same names at another width: another model.
    narrow = {k: [s[0], 16] if k.endswith("lora_A.weight") else s for k, s in lora.items()}
    other = files.lora_facts(safetensors_file(tmp_path / "b.safetensors", narrow, {}))
    assert files.matched(other.modules, modules) == 0


def test_kohya_keys_match_after_their_prefix() -> None:
    lora = files.lora_modules([("lora_unet_blocks_0_attn_to_q.lora_down.weight", [4, 32])])
    model = files.model_modules([("blocks.0.attn.to_q.weight", [64, 32])])
    assert lora == {"blocks_0_attn_to_q": 32}
    assert files.matched(lora, model) == 1


def test_a_phrase_stays_whole_and_a_list_is_split() -> None:
    assert files.metadata_words({"trigger_phrase": "a red, round fox"}) == (
        ("a red, round fox",),
        "trigger_phrase",
    )
    assert files.metadata_words({"trigger_words": '["ohwx", "clay"]'}) == (
        ("ohwx", "clay"),
        "trigger_words",
    )
    assert files.metadata_words({}) == ((), None)


def test_a_cards_images_are_its_widgets_else_the_repos() -> None:
    paths = {"README.md", "assets/2.png", "assets/1.png", "x.safetensors", "out/a.jpg"}
    card = {
        "widget": [{"text": "p", "output": {"url": "out/a.jpg"}}, {"output": {"url": "gone.png"}}]
    }
    assert _card_images(card, None, paths) == ("out/a.jpg",)
    assert _card_images({}, None, paths) == ("assets/1.png", "assets/2.png", "out/a.jpg")


def test_cached_snapshots_are_named_by_their_repo() -> None:
    snapshot = "C:/hf/hub/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/" + "f" * 40
    assert files.model_name(snapshot) == "Qwen-Image-2.1-Turbo"
