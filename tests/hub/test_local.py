"""Copies already on disk: the models folder's tree, cache trees under it, the HF cache."""

from pathlib import Path

import pytest

from imaginer.hub.layout import RepoFile
from imaginer.hub.local import find_copy, partial_bytes, safe_relpath

REPO = "acme/Chat-8B-GGUF"
FILES = [
    RepoFile("Q4/chat-Q4-00001-of-00002.gguf", 4),
    RepoFile("Q4/chat-Q4-00002-of-00002.gguf", 3),
]


def _put(root: Path, sizes: dict[str, int]) -> None:
    for path, size in sizes.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(b"x" * size)


def test_the_models_folder_tree_comes_first(tmp_path: Path) -> None:
    models, cache = tmp_path / "models", tmp_path / "cache"
    _put(
        models / "acme" / "Chat-8B-GGUF",
        {"Q4/chat-Q4-00001-of-00002.gguf": 4, "Q4/chat-Q4-00002-of-00002.gguf": 3},
    )
    _put(
        cache / "models--acme--Chat-8B-GGUF" / "snapshots" / "abc",
        {f.path: f.size or 0 for f in FILES},
    )
    copy = find_copy(REPO, FILES, models_dir=models, cache_dir=cache)
    assert copy is not None
    assert (copy.source, copy.path) == (
        "models_dir",
        models / "acme" / "Chat-8B-GGUF" / "Q4" / "chat-Q4-00001-of-00002.gguf",
    )


@pytest.mark.parametrize("tree", ["models--acme--Chat-8B-GGUF", "acme/models--acme--Chat-8B-GGUF"])
def test_cache_trees_kept_in_the_models_folder_count(tmp_path: Path, tree: str) -> None:
    """``hf download --cache-dir <models>`` (or one per owner) leaves this shape."""
    models = tmp_path / "models"
    _put(models / tree / "snapshots" / "abc", {f.path: f.size or 0 for f in FILES})
    copy = find_copy(REPO, FILES, models_dir=models, cache_dir=None)
    assert copy is not None and copy.source == "models_dir"
    assert copy.path == models / tree / "snapshots" / "abc" / FILES[0].path


def test_the_hf_cache_prefers_the_current_commit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    snapshots = cache / "models--acme--Chat-8B-GGUF" / "snapshots"
    _put(snapshots / "aaa", {f.path: f.size or 0 for f in FILES})
    _put(snapshots / "zzz", {f.path: f.size or 0 for f in FILES})
    copy = find_copy(REPO, FILES, models_dir=tmp_path / "models", cache_dir=cache, commit="zzz")
    assert copy is not None and (copy.source, copy.root) == ("hf_cache", snapshots / "zzz")


def test_a_copy_needs_every_file_at_its_size(tmp_path: Path) -> None:
    models = tmp_path / "models"
    root = models / "acme" / "Chat-8B-GGUF"
    _put(root, {"Q4/chat-Q4-00001-of-00002.gguf": 4})
    assert find_copy(REPO, FILES, models_dir=models, cache_dir=None) is None
    _put(root, {"Q4/chat-Q4-00002-of-00002.gguf": 2})  # truncated
    assert find_copy(REPO, FILES, models_dir=models, cache_dir=None) is None
    # What is there counts towards a resume: the whole first shard and the second's .part.
    (root / "Q4" / "chat-Q4-00002-of-00002.gguf").rename(
        root / "Q4" / "chat-Q4-00002-of-00002.gguf.part"
    )
    assert partial_bytes(REPO, FILES, models_dir=models) == 4 + 2


@pytest.mark.parametrize(
    "bad", ["../x.gguf", "/etc/passwd", "a/../../b", "C:/x.gguf", r"a\b.gguf", ""]
)
def test_listed_paths_must_stay_inside_the_repo(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe path"):
        safe_relpath(bad)
