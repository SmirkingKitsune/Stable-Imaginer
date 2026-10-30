"""sd.cpp-specific acquisition tests, using the upstream release naming matrix."""

import json
import stat
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from imaginer.cli import main
from imaginer.provisioning import llamacpp as shared
from imaginer.provisioning import sdcpp as sd

TAG = "master-1024-77b72a2"
PREFIX = "sd-master-77b72a2-bin-"
ASSETS = [
    PREFIX + "win-cpu-x64.zip",
    PREFIX + "win-cuda12-x64.zip",
    "cudart-sd-bin-win-cu12-x64.zip",
    PREFIX + "win-vulkan-x64.zip",
    PREFIX + "win-rocm-10.1.0-x64.zip",
    PREFIX + "Linux-Ubuntu-24.04-x86_64.zip",
    PREFIX + "Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    PREFIX + "Linux-Ubuntu-24.04-x86_64-rocm-10.1.0.zip",
    PREFIX + "Darwin-macOS-26.6.2-arm64.zip",
]


def config(root: Path, **overrides: Any) -> sd.ProvisionConfig:
    values: dict[str, Any] = {
        "provision_dir": root,
        "platform": "windows",
        "arch": "x64",
        "accelerator": "cpu",
    }
    values.update(overrides)
    return sd.ProvisionConfig(**values).normalized()


@pytest.mark.parametrize(
    "platform,arch,accelerator,asset",
    [
        ("windows", "x64", "cpu", ASSETS[0]),
        ("windows", "x64", "cuda", ASSETS[1]),
        ("windows", "x64", "vulkan", ASSETS[3]),
        ("windows", "x64", "rocm", ASSETS[4]),
        ("linux", "x64", "cpu", ASSETS[5]),
        ("linux", "x64", "vulkan", ASSETS[6]),
        ("linux", "x64", "rocm", ASSETS[7]),
        ("macos", "arm64", "metal", ASSETS[8]),
    ],
)
def test_release_matrix(tmp_path, platform, arch, accelerator, asset):
    cfg = config(tmp_path, platform=platform, arch=arch, accelerator=accelerator)
    selected = sd.select_release_assets(ASSETS, cfg)
    assert selected[0] == asset
    assert selected[1:] == ([ASSETS[2]] if accelerator == "cuda" else [])


@pytest.mark.parametrize(
    "overrides",
    [
        {"platform": "linux", "accelerator": "cuda"},
        {"platform": "linux", "arch": "arm64"},
        {"platform": "macos", "arch": "arm64"},
        {"accelerator": "cuda", "cuda_arch": "121"},
        {"accelerator": "sycl-fp16"},
    ],
)
def test_no_incorrect_platform_or_gpu_fallback(tmp_path, overrides):
    with pytest.raises(sd.ProvisionError):
        sd.select_release_assets(ASSETS, config(tmp_path, **overrides))


def test_cuda_companion_and_driver_required(tmp_path):
    cfg = config(tmp_path, accelerator="cuda", cuda_arch="120")
    with pytest.raises(sd.ProvisionError, match="runtime archive"):
        sd.select_release_assets([ASSETS[1]], cfg)
    with pytest.raises(sd.ProvisionError, match="compatible"):
        sd.select_release_assets(ASSETS, cfg, (12, 4))
    assert sd.select_release_assets(ASSETS, cfg, (12, 8)) == ASSETS[1:3]


@pytest.mark.parametrize(
    "overrides", [{"accelerator": "openvino"}, {"install_method": "bad"}, {"build_jobs": -1}]
)
def test_config_validation(tmp_path, overrides):
    with pytest.raises(sd.ProvisionError):
        config(tmp_path, **overrides)


@pytest.mark.parametrize(
    "accelerator,flag",
    [
        ("cuda", "-DSD_CUDA=ON"),
        ("rocm", "-DSD_HIPBLAS=ON"),
        ("vulkan", "-DSD_VULKAN=ON"),
        ("metal", "-DSD_METAL=ON"),
        ("sycl-fp16", "-DGGML_SYCL_F16=ON"),
        ("cpu", "-DGGML_METAL=OFF"),
    ],
)
def test_source_plan(tmp_path, accelerator, flag):
    cfg = config(tmp_path, accelerator=accelerator, cuda_arch="120", version=TAG, build_jobs=3)
    assert isinstance(cfg, sd.ProvisionConfig)
    assert cfg.exe_name == "sd-server.exe"
    assert cfg.source_dir.name == "stable-diffusion.cpp-src"
    steps = sd.build_source_plan(cfg)
    commands = [step.argv for step in steps if step.argv]
    assert any(
        argv[:3] == ["git", "clone", "--recursive"] and sd.REPO_URL in argv for argv in commands
    )
    assert ["git", "checkout", TAG] in commands
    assert ["git", "submodule", "update", "--init", "--recursive"] in commands
    configure = next(argv for argv in commands if "-S" in argv)
    assert flag in configure
    assert "-DSD_SERVER_BUILD_FRONTEND=OFF" in configure
    assert "-DSD_BUILD_EXAMPLES=ON" in configure
    if accelerator == "cuda":
        assert "-DCMAKE_CUDA_ARCHITECTURES=120a-real" in configure
    build = next(argv for argv in commands if "--build" in argv)
    assert build[-4:] == ["--target", "sd-server", "--parallel", "3"]
    assert all("llama" not in " ".join(argv) for argv in commands)
    assert not cfg.source_dir.exists()


def test_release_install_and_reuse(tmp_path, monkeypatch):
    cfg = config(tmp_path / "runtime", accelerator="cuda", cuda_arch="120", version=TAG)
    archives = {}
    for name, contents in [
        (
            ASSETS[1],
            {"build/bin/sd-server.exe": b"server", "build/bin/stable-diffusion.dll": b"engine"},
        ),
        (ASSETS[2], {"cudart/cublas64_12.dll": b"cuda"}),
    ]:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as bundle:
            for member, content in contents.items():
                bundle.writestr(member, content)
        archives[name] = path
    release = {
        "tag_name": TAG,
        "assets": [{"name": name, "browser_download_url": name} for name in archives],
    }
    monkeypatch.setattr(sd, "fetch_releases", lambda version: [release])
    monkeypatch.setattr(
        shared, "_download", lambda url, path, emit: path.write_bytes(archives[url].read_bytes())
    )
    monkeypatch.setattr(shared, "validate_server", lambda binary: "sd-server")
    installer = sd.SdCppProvisioner(cfg)
    binary = installer.ensure_runtime()
    assert binary.read_bytes() == b"server"
    assert (binary.parent / "cublas64_12.dll").read_bytes() == b"cuda"
    assert (binary.parent / "stable-diffusion.dll").read_bytes() == b"engine"
    state = sd.read_active_runtime(cfg.provision_dir)
    assert state["engine"] == "stable-diffusion.cpp"
    assert state["version"] == TAG
    assert state["ladder"][0]["ok"]
    assert installer.build_log_path is not None
    assert installer.build_log_path.name.startswith("sdcpp-provision-")
    again = sd.SdCppProvisioner(cfg)
    assert again.ensure_runtime() == binary
    assert again.used_existing


def test_source_install_and_failed_replacement(tmp_path, monkeypatch):
    cfg = config(tmp_path, platform="linux")
    output = cfg.build_dir / "bin"
    output.mkdir(parents=True)
    (output / "sd-server").write_bytes(b"good")
    (output / "libggml.so").write_bytes(b"library")
    monkeypatch.setattr(shared, "validate_server", lambda binary: "sd-server")
    monkeypatch.setattr(shared, "_run_text", lambda argv: TAG)
    sd._install_source(cfg, lambda line, err: None)
    assert (tmp_path / "bin/sd-server").read_bytes() == b"good"
    assert (tmp_path / "bin/libggml.so").read_bytes() == b"library"
    assert sd.read_active_runtime(tmp_path)["method"] == "source"

    def reject(binary):
        raise sd.ProvisionError("bad executable")

    monkeypatch.setattr(shared, "validate_server", reject)
    (output / "sd-server").write_bytes(b"bad")
    with pytest.raises(sd.ProvisionError):
        sd._install_source(cfg, lambda line, err: None)
    assert (tmp_path / "bin/sd-server").read_bytes() == b"good"


@pytest.mark.parametrize(
    "target,valid", [("libggml.so.1", True), ("../../../escape", False), ("/escape", False)]
)
def test_unix_zip_library_symlinks(tmp_path, target, valid):
    archive = tmp_path / "libs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        link = zipfile.ZipInfo("bin/libggml.so")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(link, target)
        bundle.writestr("bin/libggml.so.1", b"library")
    if valid:
        sd._extract_release(archive, tmp_path / "stage", lambda line, err: None)
        assert (tmp_path / "stage/bin/libggml.so").read_bytes() == b"library"
    else:
        with pytest.raises(sd.ProvisionError):
            sd._extract_release(archive, tmp_path / "stage", lambda line, err: None)


def test_fallback_and_cancellation(tmp_path, monkeypatch):
    cfg = config(tmp_path, platform="linux", accelerator="cuda", cuda_arch="86")
    installer = sd.SdCppProvisioner(cfg)
    attempts = installer.install_attempts()
    assert [label for label, _ in attempts] == ["source fallback", "cpu fallback"]
    calls = []

    def run(plan):
        calls.append(plan)
        if len(calls) == 1:
            raise sd.ProvisionError("missing toolkit")
        binary = tmp_path / "bin/sd-server"
        binary.parent.mkdir()
        binary.write_bytes(b"cpu")
        sd._write_state(replace(cfg, accelerator="cpu", cuda_arch=""), TAG, "release")

    monkeypatch.setattr(installer, "_run_plan", run)
    installer.ensure_runtime()
    assert "will not use the GPU" in installer.target_mismatch
    state = sd.read_active_runtime(tmp_path)
    assert state["accelerator"] == "cpu"
    assert state["ladder"][0]["skipped"]
    assert state["ladder"][-1]["ok"]
    cancelled = sd.SdCppProvisioner(cfg, cancel_check=lambda: True)
    with pytest.raises(sd.ProvisionCancelled, match=r"stable-diffusion\.cpp"):
        cancelled.ensure_runtime()


@pytest.mark.parametrize(
    "installed,method,accelerator,expected",
    [
        (TAG, "auto", "cpu", "none"),
        (TAG + "-2-gabcdef0", "auto", "cpu", "none"),
        ("master-1020-aaaaaaa-5-gabcdef0", "auto", "cpu", "none"),
        ("master-1000-aaaaaaa", "auto", "cpu", "release"),
        ("master-1000-aaaaaaa", "source", "cpu", "compile"),
        ("master-1000-aaaaaaa", "release", "sycl-fp32", "unavailable"),
        ("master-1000-aaaaaaa", "auto", "sycl-fp32", "compile"),
    ],
)
def test_update_semantics(tmp_path, monkeypatch, installed, method, accelerator, expected):
    monkeypatch.setattr(
        sd, "fetch_releases", lambda: [{"tag_name": TAG, "assets": [{"name": n} for n in ASSETS]}]
    )
    status = sd.check_for_update(
        config(tmp_path, install_method=method, accelerator=accelerator), installed
    )
    assert status.ok
    assert status.action == expected
    assert status.update_available == (expected != "none")


def test_cli_and_engine_isolation(tmp_path, capsys):
    assert (
        main(
            [
                "sdcpp",
                "plan",
                "--method",
                "source",
                "--accelerator",
                "cpu",
                "--cache-dir",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert any("sd-server" in (s["argv"] or []) for s in result[0]["steps"])
    assert not (tmp_path / "absent").exists()
    assert main(["sdcpp", "status", "--cache-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["installed"] is False
    (tmp_path / "active-runtime.json").write_text('{"version": "b42", "accelerator": "cpu"}')
    assert main(["sdcpp", "provision", "--accelerator", "cpu", "--cache-dir", str(tmp_path)]) == 1
    assert "different engine" in capsys.readouterr().err
