"""What makes a model fit to be an agent (src/imaginer/agents/fit.py): whether its chat
template takes tools, and the vision projector beside it."""

import struct
from pathlib import Path

import pytest

from imaginer.agents import fit

# The shapes of real templates, cut down: Qwen3-VL's offers tools; Gemma 3's doesn't.
QWEN = (
    "{%- if tools %}\n{{- '<|im_start|>system\\n# Tools\\n' }}\n"
    "{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}\n{%- endif %}\n"
    "{%- for message in messages %}{%- if message.tool_calls %}<tool_call>{%- endif %}"
    "{%- endfor %}"
)
GEMMA = (
    "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}{%- set first = "
    "messages[0]['content'] -%}{%- endif -%}{%- for message in messages -%}"
    "{{ '<start_of_turn>' + message['role'] }}{%- endfor -%}"
)


@pytest.mark.parametrize(
    ("template", "takes"),
    [
        (QWEN, True),
        (GEMMA, False),
        # "tools" only in the text it writes is no tools list.
        ("{{- 'You may use tools when asked.' }}{% for m in messages %}{% endfor %}", False),
        ("You have tools. {{ messages }}", False),
        ("{% if tools is not none %}{{ tools }}{% endif %}", True),
        ("{%- set tools = tools or [] -%}", True),
        ("", None),
        (None, None),
    ],
)
def test_a_template_takes_tools(template: str | None, takes: bool | None) -> None:
    assert fit.takes_tools(template) is takes


def test_params_label() -> None:
    assert fit.params_label(8190735360) == "8.2B"
    assert fit.params_label(27_000_000_000) == "27B"
    assert fit.params_label(350_000_000) == "350M"
    assert fit.params_label(None) is None


def _gguf(path: Path, meta: dict[str, str]) -> Path:
    """A GGUF header with string metadata, and no tensors."""
    out = bytearray(b"GGUF") + struct.pack("<IQQ", 3, 0, len(meta))
    for key, value in meta.items():
        name, text = key.encode(), value.encode()
        out += struct.pack("<Q", len(name)) + name
        out += struct.pack("<IQ", 8, len(text)) + text  # 8: a string
    path.write_bytes(bytes(out))
    return path


def test_a_model_with_its_projector_beside_it(tmp_path: Path) -> None:
    model = _gguf(
        tmp_path / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        {"general.architecture": "qwen3vl", "general.size_label": "8B",
         "tokenizer.chat_template": QWEN},
    )  # fmt: skip
    _gguf(tmp_path / "mmproj-Q8_0.gguf", {"general.architecture": "clip"})
    f16 = _gguf(tmp_path / "mmproj-F16.gguf", {"general.architecture": "clip"})

    found = fit.describe(model)
    assert found.model == model and found.gguf and not found.was_projector
    # F16 first, as llama.cpp's docs suggest.
    assert found.projector == f16 and [p.name for p in found.projectors] == [
        "mmproj-F16.gguf",
        "mmproj-Q8_0.gguf",
    ]
    assert (found.architecture, found.params, found.tools) == ("qwen3vl", "8B", True)


def test_a_dropped_projector_picks_the_model_beside_it(tmp_path: Path) -> None:
    projector = _gguf(tmp_path / "mmproj-model-f16.gguf", {"general.architecture": "clip"})
    _gguf(tmp_path / "gemma-3-12b-it-Q8_0.gguf", {"tokenizer.chat_template": GEMMA})
    q4 = _gguf(tmp_path / "gemma-3-12b-it-Q4_K_M.gguf", {"tokenizer.chat_template": GEMMA})
    (tmp_path / "gemma-3-12b-it-imatrix.gguf").write_bytes(b"GGUF")

    found = fit.describe(projector)
    assert found.was_projector and found.model == q4 and found.projector == projector
    assert [m.name for m in found.models] == [
        "gemma-3-12b-it-Q4_K_M.gguf",
        "gemma-3-12b-it-Q8_0.gguf",
    ]
    assert found.tools is False


def test_a_chat_template_file_replaces_the_models_own(tmp_path: Path) -> None:
    model = _gguf(tmp_path / "gemma-Q4_K_M.gguf", {"tokenizer.chat_template": GEMMA})
    template = tmp_path / "tool-use.jinja"
    template.write_text(QWEN, encoding="utf-8")
    assert fit.model_takes_tools(model) is False
    assert fit.model_takes_tools(model, template) is True
    assert fit.describe(model, chat_template=template).tools is True
    # Without a projector there is none to suggest.
    assert fit.describe(model).projector is None
