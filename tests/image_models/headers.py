"""Model files as their headers only: what family detection and launch checks read."""

import json
import struct
from pathlib import Path


def safetensors(path: Path, names: list[str], metadata: dict[str, str] | None = None) -> Path:
    header: dict[str, object] = {"__metadata__": metadata or {}}
    header |= {n: {"dtype": "F16", "shape": [4, 4], "data_offsets": [0, 0]} for n in names}
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    return path


def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(path: Path, names: list[str], arch: str | None = None) -> Path:
    """A GGUF v3 header: ``general.architecture`` (if given), then the tensor table."""
    meta = [("general.architecture", arch)] if arch else []
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(names), len(meta))
    for key, value in meta:
        out += _string(key) + struct.pack("<I", 8) + _string(value)
    for name in names:
        out += _string(name) + struct.pack("<I", 2) + struct.pack("<QQ", 4, 4)
        out += struct.pack("<IQ", 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    return path


# Tensor names that tell the families apart (a few of each model's).
SD1 = [
    "cond_stage_model.transformer.text_model.embeddings.position_ids",
    "model.diffusion_model.input_blocks.0.0.weight",
    "first_stage_model.decoder.conv_in.weight",
]
SDXL = [
    "conditioner.embedders.0.transformer.text_model.embeddings.position_ids",
    "conditioner.embedders.1.model.ln_final.weight",
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.label_emb.0.0.weight",
]
FLUX_DEV = ["double_blocks.0.img_attn.qkv.weight", "guidance_in.in_layer.weight"]
FLUX_SCHNELL = ["double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight"]
CHROMA = ["double_blocks.0.img_attn.qkv.weight", "distilled_guidance_layer.in_proj.weight"]
SD35_LARGE = [f"joint_blocks.{i}.x_block.attn.qkv.weight" for i in range(38)]
