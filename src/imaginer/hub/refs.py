"""References to Hub model repos, in the forms people paste.

    Qwen/Qwen3-VL-8B-Instruct-GGUF
    unsloth/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M         llama.cpp's -hf form, with a quant
    https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae
    https://huggingface.co/leejet/Qwen-Image-2.1-GGUF/blob/main/qwen_image_2.1-Q4_K.gguf
    hf.co/Qwen/Qwen-Image-2.1
    hf://models/Qwen/Qwen-Image-2.1/vae

Any host is accepted for URLs, so a mirror's links (``HF_ENDPOINT``) work too. Datasets and
Spaces are refused: engines load models.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

# Hub names: letters, digits, '-', '_' and '.', not starting with a separator.
_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}"
REPO_ID = re.compile(rf"^{_NAME}/{_NAME}$")
_BARE = re.compile(rf"^(?P<repo>{_NAME}/{_NAME})(?::(?P<quant>[A-Za-z0-9_.-]+))?$")
_HOSTS = ("huggingface.co/", "hf.co/", "www.huggingface.co/")
_NOT_MODELS = {"datasets", "spaces", "buckets", "collections", "papers", "docs"}


class RefError(ValueError):
    """Not a reference to a Hub model repo."""


@dataclass(frozen=True, slots=True)
class HubRef:
    repo_id: str
    revision: str = "main"
    path: str | None = None  # a file or folder in the repo, when the reference names one
    quant: str | None = None  # llama.cpp's ``:QUANT`` suffix

    @property
    def owner(self) -> str:
        return self.repo_id.partition("/")[0]

    @property
    def name(self) -> str:
        return self.repo_id.partition("/")[2]


def looks_like_ref(text: str) -> bool:
    """Whether a value typed into a path field is meant as a Hub reference rather than a
    local path: a URL, an ``hf://`` path, or ``owner/name:QUANT``. A bare ``owner/name``
    could be a relative path, so it doesn't count here."""
    value = text.strip()
    if value.startswith(("http://", "https://", "hf://")) or value.startswith(_HOSTS):
        return True
    match = _BARE.match(value)
    return bool(match and match["quant"])


def parse_ref(text: str) -> HubRef:
    value = text.strip()
    if not value:
        raise RefError("empty reference")
    if value.startswith("hf://"):
        return _from_segments(value.removeprefix("hf://").split("/"), hf_path=True)
    if value.startswith(_HOSTS):
        value = f"https://{value}"
    if value.startswith(("http://", "https://")):
        url = urlsplit(value)
        if url.path.startswith("/api/models/"):
            return _from_segments(url.path.removeprefix("/api/models/").split("/"))
        return _from_segments(url.path.strip("/").split("/"))
    match = _BARE.match(value)
    if match:
        return HubRef(match["repo"], quant=match["quant"])
    if "/" in value and "\\" not in value and ":" not in value:
        return _from_segments(value.split("/"))
    raise RefError(f"not a Hugging Face model reference: {text!r}")


def _from_segments(parts: list[str], *, hf_path: bool = False) -> HubRef:
    parts = [unquote(p) for p in parts if p]
    if hf_path:
        if parts and parts[0] in _NOT_MODELS:
            raise RefError(f"only model repos can be used, not {parts[0]}")
        if parts and parts[0] == "models":
            parts = parts[1:]
    elif parts and parts[0] in _NOT_MODELS:
        raise RefError(f"only model repos can be used, not {parts[0]}")
    if len(parts) < 2:
        raise RefError("a reference needs an owner and a repo name, like Qwen/Qwen-Image-2.1")
    name, quant, revision = parts[1], None, "main"
    if ":" in name:
        name, _, quant = name.partition(":")
    if hf_path and "@" in name:  # the hf:// file system spells a revision owner/name@rev
        name, _, revision = name.partition("@")
    repo_id = f"{parts[0]}/{name}"
    if not REPO_ID.match(repo_id):
        raise RefError(f"not a valid repo id: {repo_id!r}")
    rest = parts[2:]
    if rest and rest[0] in {"tree", "blob", "resolve", "raw"} and len(rest) >= 2:
        revision, rest = rest[1], rest[2:]
    path = "/".join(rest) or None
    if path is not None and any(p in {".", ".."} for p in rest):
        raise RefError("a path in a repo can't contain '.' or '..'")
    return HubRef(repo_id, revision=revision, path=path, quant=quant)
