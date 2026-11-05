"""Exercise installation boundaries with local fixtures, no downloads or compiler."""

import io
import json
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import pytest

from imaginer.cli import main
from imaginer.provisioning import llamacpp as lp

# Taken before conftest's autouse fixture swaps it for a fixed version.
REAL_DRIVER_PROBE = lp.detect_driver_cuda_version


def config(root: Path, **kwargs) -> lp.ProvisionConfig:
    return lp.ProvisionConfig(provision_dir=root, accelerator="cpu", **kwargs).normalized()


def test_release_installs_libraries_and_records_state(tmp_path, monkeypatch):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("package/llama-server.exe", b"server")
        bundle.writestr("package/llama.dll", b"library")
    cfg = config(tmp_path / "runtime", platform="windows", arch="x64", version="b42")
    release = {
        "tag_name": "b42",
        "assets": [{"name": "llama-b42-bin-win-cpu-x64.zip", "browser_download_url": "fixture"}],
    }
    monkeypatch.setattr(lp, "fetch_releases", lambda version: [release])
    monkeypatch.setattr(
        lp, "_download", lambda url, dest, emit: dest.write_bytes(archive.read_bytes())
    )
    monkeypatch.setattr(lp, "validate_server", lambda binary: "b42")
    installer = lp.LlamaProvisioner(cfg)
    binary = installer.ensure_runtime()
    assert binary.read_bytes() == b"server"
    assert (binary.parent / "llama.dll").read_bytes() == b"library"
    state = lp.read_active_runtime(cfg.provision_dir)
    assert state["version"] == "b42"
    assert state["ladder"][0]["ok"]
    assert lp.LlamaProvisioner(cfg).ensure_runtime() == binary


def test_download_reports_a_percentage_the_progress_bar_reads(tmp_path, monkeypatch):
    payload = b"x" * (20 * 1024**2)

    response = io.BytesIO(payload)
    response.headers = {"Content-Length": str(len(payload))}  # type: ignore[attr-defined]
    monkeypatch.setattr(lp, "urlopen", lambda request, timeout: response)
    lines: list[str] = []
    lp._download("https://example.invalid/a.zip", tmp_path / "a.zip", lambda t, e: lines.append(t))
    assert (tmp_path / "a.zip").read_bytes() == payload
    assert lines == ["[ 40%] Downloaded 8 of 20 MiB", "[ 80%] Downloaded 16 of 20 MiB"]
    assert lp.parse_progress_fraction(lines[0]) == 0.4
    assert lp._download_line(8 * 1024**2, 0) == "Downloaded 8 MiB"  # no Content-Length


def test_swap_moves_aside_a_previous_runtime_it_cannot_delete(tmp_path, monkeypatch):
    # On Windows a server still running from bin.previous keeps its files undeletable.
    for name, content in [("bin.previous", b"oldest"), ("bin", b"old"), ("bin.next", b"new")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "llama-server.exe").write_bytes(content)
    real_rmtree = shutil.rmtree

    def held(path, *args, **kwargs):
        if Path(path).name != "bin.previous":
            real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", held)
    lp._swap_staged_bin(tmp_path / "bin", tmp_path / "bin.next")
    assert (tmp_path / "bin/llama-server.exe").read_bytes() == b"new"
    assert (tmp_path / "bin.previous/llama-server.exe").read_bytes() == b"old"
    [stale] = tmp_path.glob("bin.previous.stale-*")
    assert (stale / "llama-server.exe").read_bytes() == b"oldest"

    # Once released, the copy moved aside goes with the next install.
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    (tmp_path / "bin.next").mkdir()
    (tmp_path / "bin.next/llama-server.exe").write_bytes(b"newer")
    lp._swap_staged_bin(tmp_path / "bin", tmp_path / "bin.next")
    assert (tmp_path / "bin/llama-server.exe").read_bytes() == b"newer"
    assert not list(tmp_path.glob("bin.previous.stale-*"))


@pytest.mark.parametrize(
    "header,version",
    [
        ("NVIDIA-SMI 617.42   KMD Version: 617.42   CUDA UMD Version: 13.4", (13, 4)),
        ("NVIDIA-SMI 572.16   Driver Version: 572.16   CUDA Version: 12.8", (12, 8)),
        ("NVIDIA-SMI has failed because it couldn't communicate with the driver", None),
    ],
)
def test_driver_cuda_version_reads_old_and_new_nvidia_smi_headers(monkeypatch, header, version):
    monkeypatch.setattr(lp, "_run_text", lambda argv: header)
    assert REAL_DRIVER_PROBE() == version


def test_failed_validation_preserves_previous_runtime(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    old = tmp_path / "bin" / cfg.exe_name
    old.parent.mkdir()
    old.write_bytes(b"old")
    built = cfg.build_dir / "bin" / cfg.exe_name
    built.parent.mkdir(parents=True)
    built.write_bytes(b"bad")

    def reject(binary):
        raise lp.ProvisionError("cannot load library")

    monkeypatch.setattr(lp, "validate_server", reject)
    with pytest.raises(lp.ProvisionError, match="cannot load"):
        lp._install_source_build(cfg, lambda line, err: None)
    assert old.read_bytes() == b"old"


def test_source_install_copies_libraries(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    built = cfg.build_dir / "bin" / cfg.exe_name
    built.parent.mkdir(parents=True)
    built.write_bytes(b"server")
    (built.parent / "ggml.dll").write_bytes(b"library")
    monkeypatch.setattr(lp, "validate_server", lambda binary: "b42")
    monkeypatch.setattr(lp, "_run_text", lambda argv: "b42")
    lp._install_source_build(cfg, lambda line, err: None)
    assert (tmp_path / "bin" / "ggml.dll").is_file()
    assert lp.read_active_runtime(tmp_path)["method"] == "source"


def test_pinned_version_does_not_reuse_other_version(tmp_path, monkeypatch):
    cfg = config(tmp_path, version="b43")
    binary = tmp_path / "bin" / cfg.exe_name
    binary.parent.mkdir()
    binary.write_bytes(b"old")
    lp._write_version(config(tmp_path, version="b42"), binary, "b42", "release")
    installer = lp.LlamaProvisioner(cfg)

    def fail(plan):
        raise lp.ProvisionError("new install attempted")

    monkeypatch.setattr(installer, "_run_plan", fail)
    with pytest.raises(lp.ProvisionError, match="new install attempted"):
        installer.ensure_runtime()
    assert not installer.used_existing
    assert binary.read_bytes() == b"old"


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:/escape", "..\\escape", "a:stream"])
def test_archive_paths_cannot_escape(tmp_path, name):
    with pytest.raises(lp.ProvisionError):
        lp._safe_archive_target(tmp_path, name)


def test_tar_rejects_external_symlink(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        link = tarfile.TarInfo("link")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        bundle.addfile(link)
        item = tarfile.TarInfo("link/file")
        item.size = 1
        bundle.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(tarfile.FilterError):
        lp._extract(archive, tmp_path / "stage")
    assert not (tmp_path.parent / "outside").exists()


def test_install_lock_is_exclusive_and_released(tmp_path):
    with (
        lp._install_lock(tmp_path),
        pytest.raises(lp.ProvisionError, match="Another installation"),
        lp._install_lock(tmp_path),
    ):
        pytest.fail("second lock acquired")
    with lp._install_lock(tmp_path):
        pass


def test_cancel_silent_process(tmp_path):
    started = time.monotonic()
    installer = lp.LlamaProvisioner(
        config(tmp_path), cancel_check=lambda: time.monotonic() - started > 0.3
    )
    step = lp.InstallStep("silent", argv=[sys.executable, "-c", "import time; time.sleep(30)"])
    with pytest.raises(lp.ProvisionCancelled):
        installer._run_command(step, lp.StepProgress(1, 1, "silent"))
    assert time.monotonic() - started < 10


def test_cli_plan_is_json_and_does_not_install(tmp_path, capsys):
    root = tmp_path / "absent"
    assert (
        main(
            [
                "llama",
                "plan",
                "--method",
                "source",
                "--accelerator",
                "cpu",
                "--cache-dir",
                str(root),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result[0]["attempt"] == "source"
    assert any("--build" in (step["argv"] or []) for step in result[0]["steps"])
    assert not root.exists()


def test_cli_status_and_failure(tmp_path, capsys, monkeypatch):
    assert main(["llama", "status", "--cache-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["installed"] is False

    def fail(self):
        raise lp.ProvisionError("test failure")

    monkeypatch.setattr(lp.LlamaProvisioner, "ensure_runtime", fail)
    assert main(["llama", "provision", "--cache-dir", str(tmp_path), "--accelerator", "cpu"]) == 1
    assert "test failure" in capsys.readouterr().err


def test_blackwell_release_needs_compatible_cuda_toolkit(tmp_path):
    cfg = lp.ProvisionConfig(
        provision_dir=tmp_path,
        accelerator="cuda",
        cuda_arch="120",
        platform="linux",
        arch="x64",
    )
    names = [
        "llama-b42-bin-ubuntu-cuda-12.4-x64.tar.gz",
        "llama-b42-bin-ubuntu-cuda-13.4-x64.tar.gz",
    ]
    assert lp.select_release_assets(names, cfg) == names[1:]
    with pytest.raises(lp.ProvisionError):
        lp.select_release_assets(names, cfg, (12, 4))


def test_fallback_records_actual_accelerator(tmp_path, monkeypatch):
    cfg = lp.ProvisionConfig(
        provision_dir=tmp_path,
        accelerator="cuda",
        cuda_arch="86",
        platform="linux",
        arch="x64",
    )
    installer = lp.LlamaProvisioner(cfg)
    count = 0

    def run(plan):
        nonlocal count
        count += 1
        if count < 3:
            raise lp.ProvisionError("GPU unavailable")
        binary = tmp_path / "bin" / cfg.exe_name
        binary.parent.mkdir()
        binary.write_bytes(b"cpu")
        lp._write_version(config(tmp_path, platform="linux", arch="x64"), binary, "b42", "release")

    monkeypatch.setattr(installer, "_run_plan", run)
    installer.ensure_runtime()
    state = lp.read_active_runtime(tmp_path)
    assert state["accelerator"] == "cpu"
    assert [rung["ok"] for rung in state["ladder"]] == [False, False, True]
    assert "Inference will not use the GPU" in installer.target_mismatch


def test_cancellation_never_enters_cpu_fallback(tmp_path, monkeypatch):
    installer = lp.LlamaProvisioner(config(tmp_path))
    calls = []

    def cancel(plan):
        calls.append(plan)
        raise lp.ProvisionCancelled("cancelled")

    monkeypatch.setattr(installer, "_run_plan", cancel)
    with pytest.raises(lp.ProvisionCancelled):
        installer.ensure_runtime()
    assert len(calls) == 1


def test_build_logs_keep_bounded_tail(tmp_path, monkeypatch):
    from imaginer.provisioning.log_retention import RetentionPolicy

    monkeypatch.setattr(lp, "BUILD_LOG_RETENTION", RetentionPolicy(2, 2048, 1024, 128))
    installer = lp.LlamaProvisioner(config(tmp_path))
    installer._open_build_log()
    installer._log("line\n" * 300, False)
    assert installer.build_log_path is not None
    assert installer.build_log_path.stat().st_size < 1024
    assert "log truncated" in installer.build_log_path.read_text()


def test_command_failure_includes_diagnostic_tail(tmp_path):
    installer = lp.LlamaProvisioner(config(tmp_path))
    step = lp.InstallStep(
        "fail", argv=[sys.executable, "-c", "import sys; print('compiler error'); sys.exit(2)"]
    )
    with pytest.raises(lp.ProvisionError, match="compiler error"):
        installer._run_command(step, lp.StepProgress(1, 1, "fail"))
