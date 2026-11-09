"""The Hub's REST API, over httpx.

It reads what the ``hf`` CLI and huggingface_hub read, so a machine that is already logged
in needs nothing more:

* **Token:** ``$HF_TOKEN`` (or the env var ``hub.token_env`` names), else the file
  ``hf auth login`` writes: ``$HF_TOKEN_PATH``, else ``$HF_HOME/token``.
* **Endpoint:** ``hub.endpoint``, else ``$HF_ENDPOINT`` (a mirror), else https://huggingface.co.

The token is read for each request, so logging in takes effect without a restart. It goes
to the endpoint only: httpx drops it when a download redirects to the CDN, and it is never
logged or returned by the API.
"""

import json
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from imaginer import __version__
from imaginer.domain.errors import HubAccessError, HubError, NotFoundError
from imaginer.hub.layout import RepoFile
from imaginer.hub.local import safe_relpath

DEFAULT_ENDPOINT = "https://huggingface.co"
# Fields a search result carries (without expand[], the Hub leaves out gated).
_EXPAND = ("downloads", "likes", "gated", "private", "pipeline_tag", "library_name",
           "lastModified", "tags")  # fmt: skip
MAX_TEXT_BYTES = 1024 * 1024


def hf_home() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return Path(os.path.expandvars(os.environ.get("HF_HOME") or default)).expanduser()


def hub_cache_dir() -> Path:
    """Where huggingface_hub (and so Diffusers and transformers) keeps downloads."""
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(os.path.expandvars(configured)).expanduser()
    return hf_home() / "hub"


def resolve_endpoint(configured: str | None) -> str:
    return (configured or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")


@dataclass(frozen=True, slots=True)
class Credentials:
    token: str | None
    source: str | None  # where it came from, for the UI: "$HF_TOKEN" or "hf auth login"


def find_token(token_env: str = "HF_TOKEN") -> Credentials:
    for name in dict.fromkeys((token_env, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
        value = os.environ.get(name, "").strip()
        if value:
            return Credentials(value, f"${name}")
    path = Path(os.environ.get("HF_TOKEN_PATH") or hf_home() / "token").expanduser()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    return Credentials(value, "hf auth login") if value else Credentials(None, None)


@dataclass(frozen=True, slots=True)
class RepoSummary:
    id: str
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library: str | None = None
    gated: bool = False
    private: bool = False
    updated_at: str | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RepoInfo:
    summary: RepoSummary
    sha: str | None
    files: tuple[RepoFile, ...]
    base_models: tuple[str, ...] = ()
    instance_prompt: str | None = None  # a LoRA card's trigger words
    images: tuple[str, ...] = ()  # the card's example images, as paths in the repo


class HubClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        token: Callable[[], str | None] = lambda: None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._token = token
        self._http = httpx.AsyncClient(
            transport=transport,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0, read=60.0),
            headers={"User-Agent": f"stable-imaginer/{__version__}"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def _headers(self) -> dict[str, str]:
        token = self._token()
        return {"Authorization": f"Bearer {token}"} if token else {}

    def repo_url(self, repo_id: str) -> str:
        return f"{self.endpoint}/{repo_id}"

    def file_url(self, repo_id: str, revision: str, path: str) -> str:
        return f"{self.endpoint}/{repo_id}/resolve/{quote(revision, safe='')}/{quote(path)}"

    async def search(
        self,
        query: str = "",
        *,
        filters: Sequence[str] = (),
        pipeline_tag: str | None = None,
        sort: str = "downloads",
        limit: int = 30,
    ) -> list[RepoSummary]:
        params: list[tuple[str, str | int]] = [("sort", sort), ("direction", -1), ("limit", limit)]
        if query:
            params.append(("search", query))
        if pipeline_tag:
            params.append(("pipeline_tag", pipeline_tag))
        params += [("filter", f) for f in filters]
        params += [("expand[]", field) for field in _EXPAND]
        data = await self._json(f"{self.endpoint}/api/models", params=params, what="the search")
        if not isinstance(data, list):
            raise HubError("the Hub's search answered with something other than a list")
        return [_summary(item) for item in data if isinstance(item, dict) and "id" in item]

    async def repo(self, repo_id: str, revision: str = "main") -> RepoInfo:
        url = f"{self.endpoint}/api/models/{repo_id}/revision/{quote(revision, safe='')}"
        data = await self._json(url, params={"blobs": "true"}, what=repo_id)
        if not isinstance(data, dict):
            raise HubError(f"the Hub's answer for {repo_id} is not an object")
        files = []
        for sibling in data.get("siblings") or []:
            name = sibling.get("rfilename")
            if not isinstance(name, str) or not _safe(name):
                continue  # never joined to a local folder, so never downloaded or looked for
            lfs = sibling.get("lfs") or {}
            files.append(RepoFile(name, _int(sibling.get("size")), lfs.get("sha256") or None))
        card = data.get("cardData") or {}
        base = card.get("base_model") if isinstance(card, dict) else None
        bases = (
            (base,) if isinstance(base, str) else tuple(b for b in base or () if isinstance(b, str))
        )
        prompt = card.get("instance_prompt") if isinstance(card, dict) else None
        if isinstance(prompt, list):
            prompt = ", ".join(str(p) for p in prompt if isinstance(p, str))
        return RepoInfo(
            _summary(data),
            data.get("sha"),
            tuple(files),
            bases,
            instance_prompt=prompt.strip()[:500] or None if isinstance(prompt, str) else None,
            images=_card_images(card, data.get("widgetData"), {f.path for f in files}),
        )

    async def json_file(self, repo_id: str, revision: str, path: str) -> Any:
        """A small JSON file from the repo, such as model_index.json."""
        async with self.open(repo_id, revision, path) as resp:
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body += chunk
                if len(body) > MAX_TEXT_BYTES:
                    raise HubError(f"{path} in {repo_id} is larger than expected")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise HubError(f"{path} in {repo_id} is not valid JSON") from exc

    @asynccontextmanager
    async def open(
        self, repo_id: str, revision: str, path: str, *, start: int = 0
    ) -> AsyncIterator[httpx.Response]:
        """Stream one file, from byte ``start`` on. A 206 answer resumes; a 200 one starts
        over (the server ignored the range), and a 416 one means nothing is left."""
        headers = self._headers()
        if start:
            headers["Range"] = f"bytes={start}-"
        request = self._http.build_request(
            "GET", self.file_url(repo_id, revision, path), headers=headers
        )
        try:
            resp = await self._http.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise HubError(f"couldn't reach {self.endpoint}: {_reason(exc)}") from exc
        try:
            if resp.status_code != 416:
                await _raise_for(resp, f"{repo_id}/{path}")
            yield resp
        finally:
            await resp.aclose()

    async def _json(self, url: str, *, params: Any, what: str) -> Any:
        try:
            resp = await self._http.get(url, params=params, headers=self._headers())
        except httpx.HTTPError as exc:
            raise HubError(f"couldn't reach {self.endpoint}: {_reason(exc)}") from exc
        await _raise_for(resp, what)
        try:
            return resp.json()
        except ValueError as exc:
            raise HubError(f"the Hub's answer for {what} is not JSON") from exc


async def _raise_for(resp: httpx.Response, what: str) -> None:
    if resp.is_success:
        return
    code = resp.headers.get("X-Error-Code", "")
    message = resp.headers.get("X-Error-Message", "")
    if not message:
        await resp.aread()
        message = _error_text(resp) or f"HTTP {resp.status_code}"
    if code == "GatedRepo":
        raise HubAccessError(
            f"{what} is gated: accept its terms on the Hub, then log in here "
            "(set HF_TOKEN, or run `hf auth login`)"
        )
    if code in {"RepoNotFound", "RevisionNotFound", "EntryNotFound"} or resp.status_code == 404:
        hint = "" if resp.status_code == 404 else " (or it is private: set a Hub token)"
        raise NotFoundError(f"{what} was not found on the Hub{hint}")
    if resp.status_code in {401, 403}:
        raise HubAccessError(f"the Hub refused access to {what}: {message}")
    raise HubError(f"the Hub answered {resp.status_code} for {what}: {message}")


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:200].strip()
    if isinstance(body, dict) and isinstance(body.get("error"), str):
        return str(body["error"])
    return ""


def _reason(exc: httpx.HTTPError) -> str:
    return str(exc) or type(exc).__name__


_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")


def _card_images(card: Any, widget_data: Any, paths: set[str]) -> tuple[str, ...]:
    """The example images a model card shows (its widget's outputs) that are files in the
    repo, in order; without any, the repo's image files (most cards show those)."""
    widgets = card.get("widget") if isinstance(card, dict) else None
    found = []
    for item in [*(widgets if isinstance(widgets, list) else []), *(widget_data or [])]:
        output = item.get("output") if isinstance(item, dict) else None
        url = output.get("url") if isinstance(output, dict) else None
        if isinstance(url, str) and url in paths and _safe(url) and url not in found:
            found.append(url)
    if not found:
        found = sorted(p for p in paths if p.lower().endswith(_IMAGE_TYPES) and _safe(p))
    return tuple(found[:12])


def _summary(item: dict[str, Any]) -> RepoSummary:
    gated = item.get("gated")
    return RepoSummary(
        id=str(item["id"]),
        downloads=_int(item.get("downloads")),
        likes=_int(item.get("likes")),
        pipeline_tag=item.get("pipeline_tag"),
        library=item.get("library_name"),
        gated=bool(gated) and gated != "false",  # False, or "auto"/"manual" when gated
        private=bool(item.get("private")),
        updated_at=item.get("lastModified"),
        tags=tuple(t for t in item.get("tags") or () if isinstance(t, str)),
    )


def _safe(path: str) -> bool:
    try:
        safe_relpath(path)
    except ValueError:
        return False
    return True


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
