"""What people paste: repo ids, Hub URLs, hf:// paths and llama.cpp's repo:QUANT."""

import pytest

from imaginer.hub.refs import HubRef, RefError, looks_like_ref, parse_ref


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Qwen/Qwen3-VL-8B-Instruct-GGUF", HubRef("Qwen/Qwen3-VL-8B-Instruct-GGUF")),
        (
            "unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M",
            HubRef("unsloth/Qwen3-VL-8B-Instruct-GGUF", quant="Q4_K_M"),
        ),
        (
            "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae",
            HubRef("Comfy-Org/Qwen-Image-2.1", path="vae"),
        ),
        (
            "https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/blob/main/qwen_image_2.1-Q4_K.gguf",
            HubRef("leejet/Qwen-Image-2.1-GGUF", path="qwen_image_2.1-Q4_K.gguf"),
        ),
        (
            "https://huggingface.co/acme/x/resolve/refs%2Fpr%2F3/a%20b.gguf?download=true",
            HubRef("acme/x", revision="refs/pr/3", path="a b.gguf"),
        ),
        ("hf.co/Qwen/Qwen-Image-2.1", HubRef("Qwen/Qwen-Image-2.1")),
        ("  https://hf-mirror.com/Qwen/Qwen-Image-2.1/  ", HubRef("Qwen/Qwen-Image-2.1")),
        ("hf://models/Qwen/Qwen-Image-2.1/vae", HubRef("Qwen/Qwen-Image-2.1", path="vae")),
        ("hf://Qwen/Qwen-Image-2.1@v2/vae", HubRef("Qwen/Qwen-Image-2.1", "v2", "vae")),
        ("https://huggingface.co/api/models/acme/x", HubRef("acme/x")),
    ],
)
def test_references_people_paste(text: str, expected: HubRef) -> None:
    assert parse_ref(text) == expected


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "empty"),
        ("https://huggingface.co/datasets/acme/data", "only model repos"),
        ("hf://spaces/acme/demo", "only model repos"),
        ("https://huggingface.co/gpt2", "owner and a repo name"),
        (r"D:\models\x.gguf", "not a Hugging Face model reference"),
        ("acme/x/../../etc", "'.' or '..'"),
        ("-acme/x", "not a valid repo id"),
    ],
)
def test_what_is_not_a_reference(text: str, fragment: str) -> None:
    with pytest.raises(RefError, match=fragment.replace("(", r"\(").replace(".", r"\.")):
        parse_ref(text)


@pytest.mark.parametrize(
    ("text", "ref"),
    [
        ("unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M", True),
        ("https://huggingface.co/acme/x", True),
        ("hf.co/acme/x", True),
        ("hf://models/acme/x", True),
        ("acme/x", False),  # could be a relative path
        ("models/x.gguf", False),
        (r"D:\models\x.gguf", False),
        ("/home/me/x.gguf", False),
    ],
)
def test_path_fields_tell_references_from_paths(text: str, ref: bool) -> None:
    assert looks_like_ref(text) is ref
