"""A fake Hugging Face Hub: the REST endpoints ``imaginer.hub.client`` uses, over real HTTP.

* ``GET /api/models``: search, by ``search`` (in the id), repeated ``filter`` (tags) and
  ``pipeline_tag``.
* ``GET /api/models/{owner}/{name}/revision/{rev}?blobs=true``: model info with siblings,
  their sizes and LFS SHA-256s.
* ``GET /{owner}/{name}/resolve/{rev}/{path}``: a file, honouring ``Range`` (206, or 416 past
  the end), with ``X-Repo-Commit``. Model weights redirect to ``/cdn/...`` first, as the real
  Hub sends LFS files to its CDN.
* Gated repos answer ``X-Error-Code: GatedRepo`` without the right token, unknown repos
  ``RepoNotFound`` and unknown files ``EntryNotFound``, in headers as the Hub does.

A file can be served corrupted (bytes that don't match its SHA-256), cut off once after some
bytes (the connection drops), or slowly, to exercise checksums, resumes and cancelling.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.routing import Route

TOKEN = "hf_fake_token"
SHA = "c0ffee" + "0" * 34
LFS_SUFFIXES = (".gguf", ".safetensors")


def weights(size: int, seed: int = 0) -> bytes:
    """Deterministic bytes standing in for a model file."""
    return bytes((i * 31 + seed) % 251 for i in range(size))


def gguf(size: int, seed: int = 0) -> bytes:
    """Stand-in GGUF: the magic the Engines screen sniffs, then filler."""
    return b"GGUF" + weights(size - 4, seed)


@dataclass
class FakeRepo:
    files: dict[str, bytes]
    pipeline_tag: str | None = None
    tags: list[str] = field(default_factory=list)
    gated: bool = False
    sha: str = SHA
    base_model: str | None = None
    downloads: int = 1000
    likes: int = 10


@dataclass
class FakeHub:
    repos: dict[str, FakeRepo] = field(default_factory=dict)
    corrupt: set[str] = field(default_factory=set)  # "repo/path": served with wrong bytes
    drop_once: dict[str, int] = field(default_factory=dict)  # "repo/path": cut after n bytes
    slow: dict[str, float] = field(default_factory=dict)  # "repo/path": seconds per 1 KB
    requests: list[dict[str, Any]] = field(default_factory=list)  # file requests, in order

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/api/models", self._search),
                Route("/api/models/{owner}/{name}/revision/{rev}", self._info),
                Route("/{owner}/{name}/resolve/{rev}/{path:path}", self._resolve),
                Route("/cdn/{owner}/{name}/{rev}/{path:path}", self._cdn),
            ]
        )

    # --- metadata -----------------------------------------------------------------------

    async def _search(self, request: Request) -> Response:
        q = request.query_params.get("search", "").lower()
        filters = request.query_params.getlist("filter")
        task = request.query_params.get("pipeline_tag")
        limit = int(request.query_params.get("limit", 30))
        found: list[dict[str, Any]] = []
        for repo_id, repo in self.repos.items():
            tags = self._tags(repo)
            if q and q not in repo_id.lower():
                continue
            if any(f not in tags for f in filters) or (task and repo.pipeline_tag != task):
                continue
            found.append(
                {
                    "_id": repo_id,
                    "id": repo_id,
                    "downloads": repo.downloads,
                    "likes": repo.likes,
                    "gated": "manual" if repo.gated else False,
                    "private": False,
                    "pipeline_tag": repo.pipeline_tag,
                    "library_name": None,
                    "lastModified": "2026-10-01T00:00:00.000Z",
                    "tags": tags,
                }
            )
        found.sort(key=lambda r: -r["downloads"])
        return JSONResponse(found[:limit])

    def _tags(self, repo: FakeRepo) -> list[str]:
        tags = list(repo.tags)
        if repo.base_model:
            tags += [f"base_model:{repo.base_model}", f"base_model:quantized:{repo.base_model}"]
        return tags

    async def _info(self, request: Request) -> Response:
        repo_id = f"{request.path_params['owner']}/{request.path_params['name']}"
        repo = self.repos.get(repo_id)
        if repo is None:
            return _error(401, "RepoNotFound", "Repository not found")
        siblings = []
        for path, data in repo.files.items():
            sibling: dict[str, Any] = {"rfilename": path, "size": len(data)}
            if path.endswith(LFS_SUFFIXES):
                digest = hashlib.sha256(data).hexdigest()
                sibling["lfs"] = {"sha256": digest, "size": len(data), "pointerSize": 130}
            siblings.append(sibling)
        return JSONResponse(
            {
                "id": repo_id,
                "sha": repo.sha,
                "pipeline_tag": repo.pipeline_tag,
                "tags": self._tags(repo),
                "gated": "manual" if repo.gated else False,
                "downloads": repo.downloads,
                "likes": repo.likes,
                "siblings": siblings,
                "cardData": {"base_model": repo.base_model} if repo.base_model else {},
            }
        )

    # --- files ------------------------------------------------------------------------------

    async def _resolve(self, request: Request) -> Response:
        p = request.path_params
        repo_id, path = f"{p['owner']}/{p['name']}", p["path"]
        found = self._find(request, repo_id, path)
        if isinstance(found, Response):
            return found
        if path.endswith(LFS_SUFFIXES):
            sha = self.repos[repo_id].sha
            return RedirectResponse(f"/cdn/{repo_id}/{sha}/{path}?sig=x", status_code=302)
        return self._serve(request, repo_id, path, found)

    async def _cdn(self, request: Request) -> Response:
        p = request.path_params
        repo_id, path = f"{p['owner']}/{p['name']}", p["path"]
        return self._serve(request, repo_id, path, self.repos[repo_id].files[path])

    def _find(self, request: Request, repo_id: str, path: str) -> bytes | Response:
        repo = self.repos.get(repo_id)
        if repo is None:
            return _error(401, "RepoNotFound", "Repository not found")
        if repo.gated:
            sent = request.headers.get("authorization")
            if sent is None:
                return _error(401, "GatedRepo", "Access to this model is restricted")
            if sent != f"Bearer {TOKEN}":
                return _error(403, "GatedRepo", "Access to this model is restricted")
        if path not in repo.files:
            return _error(404, "EntryNotFound", "Entry not found")
        return repo.files[path]

    def _serve(self, request: Request, repo_id: str, path: str, data: bytes) -> Response:
        key = f"{repo_id}/{path}"
        sent_range = request.headers.get("range")
        self.requests.append(
            {"path": key, "range": sent_range, "auth": request.headers.get("authorization")}
        )
        if key in self.corrupt:
            data = bytes(b ^ 0xFF for b in data)
        start = 0
        if sent_range and sent_range.startswith("bytes="):
            start = int(sent_range.removeprefix("bytes=").partition("-")[0])
            if start >= len(data):
                return Response(status_code=416, headers={"Content-Range": f"bytes */{len(data)}"})
        body = data[start:]
        headers = {"X-Repo-Commit": self.repos[repo_id].sha, "Accept-Ranges": "bytes"}
        status = 200
        if start:
            status = 206
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        drop = self.drop_once.pop(key, None)
        delay = self.slow.get(key)
        if drop is None and delay is None:
            return Response(body, status_code=status, headers=headers)
        headers["Content-Length"] = str(len(body))

        async def stream() -> AsyncIterator[bytes]:
            for at in range(0, len(body), 1024):
                if drop is not None and at >= drop:
                    raise ConnectionResetError("fake hub dropped the connection")
                if delay:
                    await asyncio.sleep(delay)
                yield body[at : at + 1024]

        return StreamingResponse(stream(), status_code=status, headers=headers)


def _error(status: int, code: str, message: str) -> Response:
    return JSONResponse(
        {"error": message},
        status_code=status,
        headers={"X-Error-Code": code, "X-Error-Message": message},
    )
