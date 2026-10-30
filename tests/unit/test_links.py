"""The pure parts of linking: config rules, pairing proofs, gallery merging, fit, discovery."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from imaginer.core.config import (
    LinksConfig,
    LinkShare,
    RoutingConfig,
    ServerConfig,
    Settings,
)
from imaginer.domain import models as dto
from imaginer.domain.enums import BackendKind, BackendStatus
from imaginer.engines import gpu
from imaginer.links import fit, gallery, pairing
from imaginer.links.discovery import parse
from tests.conftest import llama_config, sd_config
from tests.links.conftest import linked


def _settings(tmp_path: Path, **fields: object) -> Settings:
    return Settings(data_root=tmp_path, **fields)  # type: ignore[arg-type]


# --- config ------------------------------------------------------------------------------


def test_linked_backends_route_by_their_kind(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        backends={"nas": linked("nas", "llama", BackendKind.LLM)},
        routing=RoutingConfig(chat="nas"),
    )
    assert settings.default_backend("chat") == "nas"
    with pytest.raises(ValidationError, match="needs an image backend"):
        _settings(
            tmp_path,
            backends={"nas": linked("nas", "llama", BackendKind.LLM)},
            routing=RoutingConfig(t2i="nas"),
        )


def test_linking_beyond_this_machine_needs_a_token(tmp_path: Path) -> None:
    lan = ServerConfig(host="0.0.0.0")
    with pytest.raises(ValidationError, match="needs a server token"):
        _settings(tmp_path, server=lan, links=LinksConfig(enabled=True))
    with pytest.raises(ValidationError, match=r"can't reach 127\.0\.0\.1"):
        _settings(tmp_path, links=LinksConfig(enabled=True))
    behind_proxy = LinksConfig(enabled=True, advertise_url="https://imaginer.lan")
    with pytest.raises(ValidationError, match="needs a server token"):
        _settings(tmp_path, links=behind_proxy)
    # Two instances on one machine (development) need no token.
    local = LinksConfig(enabled=True, advertise_url="http://127.0.0.1:8701")
    assert _settings(tmp_path, links=local).links.enabled
    token = ServerConfig(host="0.0.0.0", auth_token="t")  # type: ignore[arg-type]
    assert _settings(tmp_path, server=token, links=LinksConfig(enabled=True)).links.enabled


def test_only_local_engines_can_be_shared(tmp_path: Path) -> None:
    backends = {
        "llama": llama_config("http://127.0.0.1:1"),
        "remote": linked("nas", "sd", BackendKind.IMAGE),
    }
    with pytest.raises(ValidationError, match="already runs on a linked instance"):
        _settings(tmp_path, backends=backends, links=LinksConfig(share={"remote": LinkShare()}))
    with pytest.raises(ValidationError, match="unknown backend 'nope'"):
        _settings(tmp_path, backends=backends, links=LinksConfig(share={"nope": LinkShare()}))
    # null un-shares what a profile shares (the .local.yaml sidecar writes that)
    unshared = LinksConfig(share={"llama": None, "nope": None})
    assert _settings(tmp_path, backends=backends, links=unshared).links.shared() == {}


# --- pairing -----------------------------------------------------------------------------


def test_codes_survive_retyping() -> None:
    code = pairing.new_code()
    assert len(code) == 9 and code[4] == "-"
    assert set(code.replace("-", "")) <= set(pairing.ALPHABET)
    assert pairing.code_key(code) == pairing.code_key(f" {code.lower().replace('-', ' ')} ")
    assert pairing.normalize_code("ol1i-abcd") == "0111ABCD"


def test_proofs_bind_the_challenge_and_both_instances() -> None:
    key = pairing.code_key("ABCD-EFGH")
    proof = pairing.proof(key, "challenge", "a", "b")
    assert pairing.proof_matches(proof, proof.upper())
    for other in (
        pairing.proof(key, "other challenge", "a", "b"),
        pairing.proof(key, "challenge", "b", "a"),
        pairing.proof(pairing.code_key("ABCD-EFGJ"), "challenge", "a", "b"),
    ):
        assert not pairing.proof_matches(proof, other)


# --- gallery -----------------------------------------------------------------------------


def _item(n: int, source: str) -> dto.GalleryItem:
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=n)
    return dto.GalleryItem(
        id=f"{source}-{n:02d}",
        title=None,
        created_at=stamp,
        updated_at=stamp,
        archived_at=None,
        origin=dto.Origin(instance_id=source, name=source),
    )


def _source(items: list[int], name: str) -> list[dto.GalleryItem]:
    return [_item(n, name) for n in sorted(items, reverse=True)]


def test_merging_pages_from_uneven_sources() -> None:
    a, b = _source([9, 8, 7, 1], "a"), _source([6, 5, 4, 3, 2], "b")
    seen: list[str] = []
    positions: dict[str, gallery.Position] = {}
    for _ in range(10):
        pages = {}
        for name, items in (("a", a), ("b", b)):
            if name in positions and positions[name] is None:
                continue
            after = positions.get(name)
            rest = [i for i in items if after is None or (i.updated_at.isoformat(), i.id) < after]
            pages[name] = gallery.SourcePage(items=rest[:3], has_more=len(rest) > 3)
        taken, positions = gallery.merge(pages, positions, 3)
        seen += [i.id for i in taken]
        if all(positions.get(s, ("", "")) is None for s in ("a", "b")):
            break
    assert seen == [i.id for i in sorted(a + b, key=lambda i: i.updated_at, reverse=True)]


def test_gallery_cursors_round_trip() -> None:
    positions: dict[str, gallery.Position] = {
        "local": ("2026-01-01T00:00:00+00:00", "x"),
        "l": None,
    }
    assert gallery.decode(gallery.encode(positions)) == positions
    with pytest.raises(Exception, match="invalid cursor"):
        gallery.decode("not-a-cursor!")


# --- fit ---------------------------------------------------------------------------------


def test_fit_states() -> None:
    card = gpu.GpuMemory(name="RTX", total_mb=16384, used_mb=2048)

    def state(status: BackendStatus, **kw: object) -> str:
        args: dict[str, object] = {
            "remote_start": True,
            "not_startable": None,
            "need": 8000,
            "gpu": card,
        }
        args.update(kw)
        return fit.assess(status=status, **args).state  # type: ignore[arg-type]

    assert state(BackendStatus.OK) == "ready"
    assert state(BackendStatus.DEGRADED) == "cold"
    assert state(BackendStatus.DOWN) == "cold"
    assert state(BackendStatus.DOWN, need=14000) == "unfit"
    assert state(BackendStatus.DOWN, remote_start=False) == "unavailable"
    assert state(BackendStatus.DOWN, not_startable="no runtime") == "unavailable"
    assert state(BackendStatus.DOWN, gpu=None) == "cold"  # fit unknown


def test_need_comes_from_the_model_files(tmp_path: Path) -> None:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + bytes(1024 * 1024 * 10 - 4))
    config = llama_config("http://127.0.0.1:1", launch={"model": model})
    assert fit.need_mb(config) == 12  # 10 MiB of weights, plus a fifth
    sd = sd_config("http://127.0.0.1:1", launch={"diffusion_model": model})
    assert fit.need_mb(sd) == 10 + fit.SDCPP_ACTIVATIONS_MB
    assert fit.need_mb(llama_config("http://127.0.0.1:1")) is None


def test_nvidia_smi_output() -> None:
    parsed = gpu.parse("NVIDIA GeForce RTX 5090, 32607, 1234\nNVIDIA X, 1, 1\n")
    assert parsed == gpu.GpuMemory(name="NVIDIA GeForce RTX 5090", total_mb=32607, used_mb=1234)
    assert parsed.free_mb == 32607 - 1234
    assert gpu.parse("") is None
    assert gpu.parse("N/A, [Not Supported], 1") is None


# --- discovery ---------------------------------------------------------------------------


def test_advertisements_become_instances() -> None:
    props = {b"id": b"abc", b"name": b"nas", b"v": b"1", b"url": b"http://nas.local:8700"}
    found = parse(props, ["192.168.1.20", "127.0.0.1"], 8700)
    assert found is not None
    instance, candidates = found
    assert (instance.instance_id, instance.name, instance.api_version) == ("abc", "nas", 1)
    assert candidates == ("http://nas.local:8700", "http://192.168.1.20:8700")
    # A loopback URL is useless to others: the address is used instead.
    loopback = parse({**props, b"url": b"http://127.0.0.1:8700"}, ["10.0.0.5"], 8700)
    assert loopback is not None and loopback[0].url == "http://10.0.0.5:8700"
    assert parse({b"name": b"no id"}, ["10.0.0.5"], 8700) is None
