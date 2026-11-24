"""Artifact library pieces that need no server: cursors, refs, generation details."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from imaginer.artifact_library import library
from imaginer.domain import models as dto
from imaginer.domain.enums import ArtifactKind, ArtifactSource, JobTask
from imaginer.links import gallery


def _item(n: int, source: str = "local", **extra: Any) -> dto.ArtifactLibraryItem:
    return dto.ArtifactLibraryItem(
        id=f"{n:04d}",
        ref=dto.LibraryRef(artifact_id=f"{n:04d}"),
        live=True,
        kind=ArtifactKind.IMAGE,
        source=ArtifactSource.UPLOAD,
        mime="image/png",
        sha256=f"{n:064d}",
        size_bytes=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        content_url="/x",
        origin=dto.Origin(instance_id=source, name=source),
        **extra,
    )


def test_library_cursors_merge_by_id() -> None:
    a = [_item(n, "a") for n in (9, 8, 7, 1)]
    b = [_item(n, "b") for n in (6, 5, 4, 3, 2)]
    seen: list[str] = []
    positions: dict[str, gallery.Position] = {}
    for _ in range(10):
        pages = {}
        for name, items in (("a", a), ("b", b)):
            if name in positions and positions[name] is None:
                continue
            after = positions.get(name)
            rest = [i for i in items if after is None or i.id < after[0]]
            pages[name] = gallery.SourcePage(items=rest[:3], has_more=len(rest) > 3)
        taken, positions = gallery.merge(
            pages, positions, 3, sort_key=library._by_id, position=library._id_position
        )
        seen += [i.id for i in taken]
        positions = gallery.decode(gallery.encode(positions))  # as the client sends it back
        if all(positions.get(s, ("",)) is None for s in ("a", "b")):
            break
    assert seen == [f"{n:04d}" for n in range(9, 0, -1)]


def test_cursors_reject_what_isnt_one() -> None:
    assert gallery.decode(gallery.encode({"local": ("x",), "l": None})) == {
        "local": ("x",),
        "l": None,
    }
    for bad in ("not-a-cursor!", gallery.encode({"a": ()}), "W10"):  # W10 is "[]"
        with pytest.raises(Exception, match="invalid cursor"):
            gallery.decode(bad)


def test_a_ref_names_one_image() -> None:
    assert dto.LibraryRef(pin_id="p").origin == "local"
    for bad in ({}, {"artifact_id": "a", "pin_id": "p"}):
        with pytest.raises(ValidationError):
            dto.LibraryRef.model_validate(bad)
    # A newer instance may send more: that's fine.
    assert dto.LibraryRef.model_validate({"artifact_id": "a", "shiny": True}).artifact_id == "a"


def _job(**params: Any) -> Any:
    return SimpleNamespace(
        id="job",
        task=JobTask.EDIT,
        backend="sd",
        model="qwen.gguf",
        alpha=False,
        seed=3,
        params_json=params,
    )


def test_generation_details_from_whatever_the_job_kept() -> None:
    full = library._generation(
        _job(resolved={"steps": 8, "sampler": "euler"}, model_card="qwen"), {"seed": 11}
    )
    assert full is not None
    assert (full.task, full.model_card, full.seed, full.resolved) == (
        "edit",
        "qwen",
        11,
        {"steps": 8, "sampler": "euler"},
    )
    bare = library._generation(_job(requested={"prompt": "x"}), {})
    assert bare is not None and (bare.resolved, bare.model_card, bare.seed) == (None, None, 3)
    assert library._generation(None, {"seed": 1}) is None
    assert library._quiet_generation({"seed": "not a number"}) is None


def test_an_import_records_where_it_came_from() -> None:
    item = _item(1, label="img-4", prompt="a fox", original_filename="C_fox.webp")
    item = item.model_copy(update={"mime": "image/webp", "conversation_title": "foxes"})
    meta = library._provenance(item)
    assert meta["original_filename"] == "C_fox.webp"
    assert meta["imported_from"] == {
        "instance_id": "local",
        "instance_name": "local",
        "artifact_id": "0001",
        "label": "img-4",
        "source": "upload",
        "prompt": "a fox",
        "conversation_title": "foxes",
    }
    unnamed = library._provenance(_item(2))
    assert unnamed["original_filename"] == "image.png"


def test_pin_files_stay_in_their_folder(tmp_path: Path) -> None:
    lib = library.ArtifactLibrary.__new__(library.ArtifactLibrary)
    lib._root = tmp_path
    good = "019a0000-0000-7000-8000-000000000000.png"
    assert lib._pin_path(good) == tmp_path / good
    for bad in ("../x.png", "019a0000-0000-7000-8000-000000000000.exe", "a/b.png", ""):
        assert lib._pin_path(bad) is None
