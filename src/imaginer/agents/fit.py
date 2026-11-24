"""What makes a chat model fit to be an agent.

An agent looks at the images it makes, so its model **must see images**: llama-server loads
a vision projector beside it (``--mmproj``, an ``mmproj*.gguf``). It **should call tools**:
an agent that can't call ``generate_image`` or ``edit_image`` can only talk about images.
Whether it can is in its chat template: llama-server offers tools to a model whose Jinja
template takes a ``tools`` list (``chat_template_caps.supports_tools``), so a template that
never reads ``tools`` makes no calls. A template file given with ``--chat-template-file``
replaces the model's own.

The checks here read files on this machine (blocking); the Hub's side (a repo's projectors,
the chat template of its GGUF) comes from ``HubService.agent_files``.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from imaginer.hub.layout import PREFERRED_PROJECTORS, PREFERRED_QUANTS
from imaginer.loras import files as names

MAX_TEMPLATE_BYTES = 1024 * 1024
# A Jinja statement or expression, whose string literals are dropped before looking for
# ``tools`` (a prompt that only says "tools" in its text takes none).
_BLOCK = re.compile(r"\{%-?(.*?)-?%\}|\{\{-?(.*?)-?\}\}", re.DOTALL)
_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.DOTALL)
_TOOLS = re.compile(r"\btools\b")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})$")


def takes_tools(template: str | None) -> bool | None:
    """Whether a chat template takes a ``tools`` list (so the model can call tools). None
    when there is no template to read."""
    if not template or not template.strip():
        return None
    for match in _BLOCK.finditer(template):
        code = _STRING.sub("''", match.group(1) or match.group(2) or "")
        if _TOOLS.search(code):
            return True
    return False


def is_projector(path: Path, meta: dict[str, str] | None = None) -> bool:
    """A vision projector rather than a model: named ``mmproj…``, or a GGUF whose
    architecture is ``clip`` (what llama.cpp writes for projectors)."""
    if "mmproj" in path.name.lower():
        return True
    return (meta or {}).get("general.architecture") == "clip"


def params_label(total: int | None) -> str | None:
    """A parameter count as people say it: 8190735360 -> "8.2B"."""
    if not total or total <= 0:
        return None
    if total >= 1e9:
        return f"{total / 1e9:.1f}B".replace(".0B", "B")
    return f"{round(total / 1e6)}M"


def _rank(order: tuple[str, ...]) -> Callable[[Path], tuple[int, int, str]]:
    def key(p: Path) -> tuple[int, int, str]:
        quant = names.model_quant(p.name)
        position = order.index(quant) if quant in order else len(order)
        return (position, len(p.name), p.name.lower())

    return key


def _ggufs(folder: Path) -> list[Path]:
    try:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".gguf"]
    except OSError:
        return []


def projectors_beside(model: Path) -> list[Path]:
    """The vision projectors in a model's folder, the one to suggest first (F16, BF16,
    Q8_0…)."""
    found = [p for p in _ggufs(model.parent) if "mmproj" in p.name.lower() and p != model]
    return sorted(found, key=_rank(PREFERRED_PROJECTORS))


def models_beside(path: Path) -> list[Path]:
    """The chat models in a folder (each sharded one by its first shard), Q4_K_M first."""
    found = []
    for p in _ggufs(path.parent):
        if "mmproj" in p.name.lower() or "imatrix" in p.name.lower():
            continue
        shard = _SHARD.search(p.stem)
        if shard and shard.group(1) != "00001":
            continue
        found.append(p)
    return sorted(found, key=_rank(PREFERRED_QUANTS))


def read_template(path: Path) -> str | None:
    """A chat template file's text (--chat-template-file), or None if it can't be read."""
    try:
        if path.stat().st_size > MAX_TEMPLATE_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


@dataclass(slots=True)
class LocalModel:
    """A model file on this machine, as an agent would run it."""

    model: Path | None  # None: the path was a projector and no model is beside it
    exists: bool
    gguf: bool
    projectors: list[Path] = field(default_factory=list)
    projector: Path | None = None
    models: list[Path] = field(default_factory=list)
    architecture: str | None = None
    params: str | None = None
    tools: bool | None = None
    was_projector: bool = False  # the path given is a projector; its model is picked


def describe(
    path: Path, *, mmproj: Path | None = None, chat_template: Path | None = None
) -> LocalModel:
    """A model file (or a projector, whose folder's model is then picked) with what an
    agent needs: its vision projectors and whether its chat template takes tools. Blocking."""
    meta = names.gguf_metadata(path) if path.is_file() else None
    was_projector = path.is_file() and is_projector(path, meta)
    models = models_beside(path) if was_projector else ([path] if path.is_file() else [])
    model = models[0] if models else None
    if was_projector:
        meta = names.gguf_metadata(model) if model is not None else None
    found = projectors_beside(model) if model is not None else []
    if was_projector and path not in found:
        found.insert(0, path)
    if mmproj is not None:
        found = [mmproj, *(p for p in found if p != mmproj)]
    projector = mmproj if mmproj is not None else (path if was_projector else None)
    projector = projector or (found[0] if found else None)
    template = read_template(chat_template) if chat_template is not None else None
    if template is None and meta is not None:
        template = meta.get("tokenizer.chat_template")
    return LocalModel(
        model=model,
        exists=path.exists(),
        gguf=meta is not None or (model is not None and _is_gguf(model)),
        projectors=found,
        projector=projector,
        models=models,
        architecture=(meta or {}).get("general.architecture"),
        params=(meta or {}).get("general.size_label"),
        tools=takes_tools(template),
        was_projector=was_projector,
    )


def model_takes_tools(model: Path, chat_template: Path | None = None) -> bool | None:
    """Whether the model (or the template file that replaces its own) takes tools; None
    when neither can be read. Blocking (the GGUF header is read once, then cached)."""
    if chat_template is not None:
        text = read_template(chat_template)
        if text is not None:
            return takes_tools(text)
    meta = names.gguf_metadata(model) if model.is_file() else None
    return takes_tools((meta or {}).get("tokenizer.chat_template"))


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == b"GGUF"
    except OSError:
        return False
