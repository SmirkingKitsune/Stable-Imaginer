"""Copies of a repo's files already on this machine.

Two places are looked in:

* **The models folder** (``hub.models_dir``, default ``<data_root>/models``), where this
  server downloads to: ``<models_dir>/<owner>/<name>/<path in the repo>``, the same tree as
  ``hf download --local-dir``. A ``.part`` file beside a missing one is a download to resume.
* **Hugging Face cache trees**, where ``hf download``, Diffusers and transformers put what
  they fetch: ``models--<owner>--<name>/snapshots/<commit>/<path>``. The Hugging Face cache
  itself is one; one kept in the models folder (``<models_dir>/models--...``, or per owner,
  ``<models_dir>/<owner>/models--...``, as ``hf download --cache-dir`` leaves it) counts too.
  A file found there is used in place, never copied.

A copy counts only when every file of it is present at the size the Hub lists.
"""

import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from imaginer.hub.layout import RepoFile

Source = Literal["models_dir", "hf_cache"]
PART_SUFFIX = ".part"


@dataclass(frozen=True, slots=True)
class LocalCopy:
    source: Source
    root: Path  # the folder that holds the repo's tree
    path: Path  # the first file: what an engine's flag points at


def safe_relpath(path: str) -> PurePosixPath:
    """A repo path that stays inside the folder it is joined to. The Hub never lists
    anything else, but its listing is not ours to trust."""
    p = PurePosixPath(path)
    if p.is_absolute() or not p.parts or any(part in {"", ".", ".."} for part in p.parts):
        raise ValueError(f"unsafe path in repo listing: {path!r}")
    if "\\" in path or ":" in path:
        raise ValueError(f"unsafe path in repo listing: {path!r}")
    return p


def repo_dir(models_dir: Path, repo_id: str) -> Path:
    owner, _, name = repo_id.partition("/")
    return models_dir / owner / name


def cache_repo_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def find_copy(
    repo_id: str,
    files: Sequence[RepoFile],
    *,
    models_dir: Path,
    cache_dir: Path | None,
    commit: str | None = None,
) -> LocalCopy | None:
    """The first complete copy of ``files``: in the models folder's own tree, else in a
    cache tree under it, else in the HF cache (the snapshot of ``commit`` first in each)."""
    if not files:
        return None
    root = repo_dir(models_dir, repo_id)
    if _complete(root, files):
        return LocalCopy("models_dir", root, root / safe_relpath(files[0].path))
    owner = repo_id.partition("/")[0]
    trees: list[tuple[Source, Path]] = [
        ("models_dir", cache_repo_dir(models_dir, repo_id)),
        ("models_dir", cache_repo_dir(models_dir / owner, repo_id)),
    ]
    if cache_dir is not None:
        trees.append(("hf_cache", cache_repo_dir(cache_dir, repo_id)))
    for source, tree in trees:
        for snapshot in _snapshots(tree, commit):
            if _complete(snapshot, files):
                return LocalCopy(source, snapshot, snapshot / safe_relpath(files[0].path))
    return None


def partial_bytes(repo_id: str, files: Iterable[RepoFile], *, models_dir: Path) -> int:
    """Bytes already in the models folder towards these files: whole files and ``.part``s."""
    return bytes_under(repo_dir(models_dir, repo_id), files)


def bytes_under(root: Path, files: Iterable[RepoFile]) -> int:
    total = 0
    for f in files:
        dest = root / safe_relpath(f.path)
        for candidate in (dest, dest.with_name(dest.name + PART_SUFFIX)):
            with contextlib.suppress(OSError):
                size = candidate.stat().st_size
                if candidate is dest and f.size is not None and size != f.size:
                    continue
                total += size
                break
    return total


def _complete(root: Path, files: Iterable[RepoFile]) -> bool:
    for f in files:
        try:
            size = (root / safe_relpath(f.path)).stat().st_size  # follows the cache's symlinks
        except (OSError, ValueError):
            return False
        if f.size is not None and size != f.size:
            return False
    return True


def _snapshots(repo_cache: Path, commit: str | None) -> list[Path]:
    snapshots = repo_cache / "snapshots"
    try:
        found = sorted((p for p in snapshots.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return []
    if commit is not None:
        found.sort(key=lambda p: p.name != commit)  # stable: the current commit first
    return found
