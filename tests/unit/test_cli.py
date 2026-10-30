import json
from pathlib import Path

import pytest

from imaginer.cli import main
from tests.fakes.fake_llama import FakeLlama


def _profile(tmp_path: Path) -> str:
    path = tmp_path / "p.yaml"
    path.write_text(f"data_root: {(tmp_path / 'data').as_posix()}\n", encoding="utf-8")
    return str(path)


def test_config_show_masks_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("IMAGINER_SERVER__AUTH_TOKEN", "super-secret")
    assert main(["config", "show", "--profile", _profile(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "super-secret" not in out
    assert json.loads(out)["server"]["auth_token"] == "**********"


def test_db_upgrade_and_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profile = _profile(tmp_path)
    assert main(["db", "check", "--profile", profile]) == 1
    assert "run `imaginer db upgrade` first" in capsys.readouterr().err
    assert main(["db", "upgrade", "--profile", profile]) == 0
    assert (tmp_path / "data" / "imaginer.sqlite3").is_file()
    assert main(["db", "check", "--profile", profile]) == 0
    assert "agree" in capsys.readouterr().out


def test_bad_profile_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show", "--profile", "missing", "--config-dir", str(tmp_path)]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_backends_check(
    tmp_path: Path, fake_llama: FakeLlama, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "b.yaml"
    lines = [
        f"data_root: {(tmp_path / 'data').as_posix()}",
        "backends:",
        f"  llama: {{type: llamacpp, base_url: '{fake_llama.url}'}}",
        "  sd: {type: sdcpp, base_url: 'http://127.0.0.1:1', enabled: false}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["backends", "check", "--profile", str(path)]) == 0
    out = capsys.readouterr().out
    assert "llama" in out and " ok " in out and "caps=chat,stream,tools,vision_in" in out
    assert "disabled in config" in out  # disabled backends don't fail the check

    fake_llama.loading = True
    assert main(["backends", "check", "--profile", str(path)]) == 1
    assert "model is loading" in capsys.readouterr().out
