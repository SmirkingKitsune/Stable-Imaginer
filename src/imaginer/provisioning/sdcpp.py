"""Provision stable-diffusion.cpp's HTTP server without a UI or model downloads.

Shares the tested execution, cancellation, locking and staging primitives with
the llama.cpp provisioner. Release names, CMake options and versioning are sd.cpp-specific.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from imaginer.provisioning import llamacpp as shared

ProvisionError = shared.ProvisionError
ProvisionCancelled = shared.ProvisionCancelled
InstallStep = shared.InstallStep
INSTALL_METHODS = shared.INSTALL_METHODS
ACCELERATORS = ("cuda", "rocm", "vulkan", "metal", "sycl-fp32", "sycl-fp16", "cpu")
REPO_URL = "https://github.com/leejet/stable-diffusion.cpp"
API_BASE = "https://api.github.com/repos/leejet/stable-diffusion.cpp"


@dataclass
class ProvisionConfig(shared.ProvisionConfig):
    provision_dir: Path = Path("cache/stable_diffusion_cpp")

    def normalized(self) -> ProvisionConfig:
        cfg = super().normalized()
        if cfg.accelerator not in ACCELERATORS:
            raise ProvisionError(f"Unsupported stable-diffusion.cpp accelerator: {cfg.accelerator}")
        if self.install_method not in INSTALL_METHODS:
            raise ProvisionError(f"Unsupported install method: {self.install_method}")
        if self.build_jobs < 0:
            raise ProvisionError("build_jobs must be nonnegative")
        if cfg.release_variant:
            raise ProvisionError("stable-diffusion.cpp does not support --release-variant")
        return ProvisionConfig(**vars(cfg))

    @property
    def exe_name(self) -> str:
        return "sd-server.exe" if self.platform == "windows" else "sd-server"

    @property
    def source_dir(self) -> Path:
        return self.provision_dir / "stable-diffusion.cpp-src"


def release_asset_patterns(cfg: shared.ProvisionConfig) -> list[str]:
    """Match only engine archives, never CLI-only names or the companion runtime."""
    arch = re.escape(cfg.arch)
    if cfg.platform == "windows":
        backend = {
            "cpu": "cpu",
            "cuda": r"cuda\d+",
            "vulkan": "vulkan",
            "rocm": r"rocm-[0-9.]+",
        }.get(cfg.accelerator)
        return [rf"^sd-.+-bin-win-{backend}-{arch}\.zip$"] if backend else []
    if cfg.platform == "linux":
        machine = {"x64": "x86_64", "arm64": "aarch64"}.get(cfg.arch, cfg.arch)
        suffix = {"cpu": "", "vulkan": "-vulkan", "rocm": r"-rocm-[0-9.]+"}.get(cfg.accelerator)
        return (
            [rf"^sd-.+-bin-Linux-Ubuntu-[0-9.]+-{re.escape(machine)}{suffix}\.zip$"]
            if suffix is not None
            else []
        )
    if cfg.platform == "macos" and cfg.accelerator == "metal":
        return [rf"^sd-.+-bin-Darwin-macOS-[0-9.]+-{arch}\.zip$"]
    return []


def select_release_assets(
    names: Sequence[str],
    cfg: shared.ProvisionConfig,
    driver_version: tuple[int, ...] | None = None,
) -> list[str]:
    patterns = release_asset_patterns(cfg)
    if not patterns:
        raise ProvisionError(
            f"No published release for {cfg.platform}/{cfg.arch}/{cfg.accelerator}"
        )
    matches = sorted(name for name in names if re.fullmatch(patterns[0], name))
    if cfg.accelerator == "cuda":
        # The official cuda12 build uses CUDA 12.8.1 and targets up to sm_120.
        # A future asset family requires verification before treating it as compatible.
        matches = [name for name in matches if "-cuda12-" in name]
        targets = shared.cuda_arch_targets(cfg.cuda_arch)
        if any(int(t) not in {61, 70, 75, 80, 86, 89, 90, 100, 120} for t in targets):
            matches = []
        if driver_version is not None and driver_version < (12, 8):
            matches = []
    if not matches:
        raise ProvisionError(
            f"No compatible sd-server archive for {cfg.platform}/{cfg.arch}/{cfg.accelerator}"
        )
    selected = [matches[-1]]
    if cfg.accelerator == "cuda":
        runtime = f"cudart-sd-bin-win-cu12-{cfg.arch}.zip"
        if runtime not in names:
            raise ProvisionError(f"Missing required CUDA runtime archive: {runtime}")
        selected.append(runtime)
    return selected


def fetch_releases(version: str = "latest") -> list[dict[str, Any]]:
    suffix = (
        "releases?per_page=20"
        if version == "latest"
        else f"releases/tags/{quote(version, safe='')}"
    )
    request = Request(
        f"{API_BASE}/{suffix}",
        headers={
            "User-Agent": "stable-imaginer-sdcpp-provisioner/1.0",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    if version != "latest":
        payload = [payload]
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ProvisionError("Invalid stable-diffusion.cpp release metadata")
    return [item for item in payload if not item.get("draft")]


def accelerator_cmake_flags(cfg: shared.ProvisionConfig) -> list[str]:
    flags = [
        f"-DSD_{name}=OFF"
        for name in ("CUDA", "HIPBLAS", "VULKAN", "METAL", "SYCL", "OPENCL", "MUSA")
    ]
    backend = {
        "cuda": "CUDA",
        "rocm": "HIPBLAS",
        "vulkan": "VULKAN",
        "metal": "METAL",
        "sycl-fp32": "SYCL",
        "sycl-fp16": "SYCL",
    }.get(cfg.accelerator)
    if backend:
        flags.remove(f"-DSD_{backend}=OFF")
        flags.append(f"-DSD_{backend}=ON")
    # GGML defaults to Metal on Apple even when SD_METAL is off.
    flags.append(f"-DGGML_METAL={'ON' if cfg.accelerator == 'metal' else 'OFF'}")
    if cfg.accelerator == "cuda" and cfg.cuda_arch:
        flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={shared.cuda_cmake_architectures(cfg.cuda_arch)}")
    if cfg.accelerator == "rocm":
        flags.extend(["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"])
    if cfg.accelerator.startswith("sycl"):
        flags.extend(
            [
                "-DCMAKE_C_COMPILER=icx",
                "-DCMAKE_CXX_COMPILER=icpx",
                f"-DGGML_SYCL_F16={'ON' if cfg.accelerator == 'sycl-fp16' else 'OFF'}",
            ]
        )
    return flags


def build_source_plan(cfg: shared.ProvisionConfig) -> list[InstallStep]:
    src, build = cfg.source_dir, cfg.build_dir
    ref = "master" if cfg.version == "latest" else cfg.version
    steps = []
    if not cfg.bypass_environment_checks:
        steps.append(
            InstallStep(
                "Checking source-build prerequisites",
                action=lambda emit: shared.preflight(cfg, emit),
            )
        )
    if not (src / ".git").exists():
        steps.append(
            InstallStep(
                "Cloning stable-diffusion.cpp",
                argv=["git", "clone", "--recursive", REPO_URL, str(src)],
                cwd=cfg.provision_dir,
            )
        )
    steps.extend(
        [
            InstallStep("Fetching tags", argv=["git", "fetch", "--tags", "origin"], cwd=src),
            InstallStep(f"Checking out {ref}", argv=["git", "checkout", ref], cwd=src),
        ]
    )
    if cfg.version == "latest":
        steps.append(
            InstallStep(
                "Updating source", argv=["git", "pull", "--ff-only", "origin", "master"], cwd=src
            )
        )
    steps.append(
        InstallStep(
            "Initializing submodules",
            argv=["git", "submodule", "update", "--init", "--recursive"],
            cwd=src,
        )
    )
    # Clear generated configuration so toolkit changes and accelerator overrides cannot
    # leave a stale GGML backend enabled. Compiled objects remain available to the build.
    steps.extend(
        [
            InstallStep(
                "Clearing CMake cache",
                argv=["cmake", "-E", "rm", "-f", str(build / "CMakeCache.txt")],
                cwd=src,
            ),
            InstallStep(
                "Clearing compiler selection",
                argv=["cmake", "-E", "rm", "-rf", str(build / "CMakeFiles")],
                cwd=src,
            ),
        ]
    )
    configure = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSD_BUILD_EXAMPLES=ON",
        "-DSD_SERVER_BUILD_FRONTEND=OFF",
        *accelerator_cmake_flags(cfg),
        *cfg.cmake_args,
    ]
    if cfg.platform != "windows":
        configure.extend(
            [
                "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
                "-DCMAKE_INSTALL_RPATH="
                + ("@loader_path" if cfg.platform == "macos" else "$ORIGIN"),
            ]
        )
    steps.extend(
        [
            InstallStep("Configuring stable-diffusion.cpp", argv=configure, cwd=src),
            InstallStep(
                "Building sd-server",
                argv=[
                    "cmake",
                    "--build",
                    str(build),
                    "--config",
                    "Release",
                    "--target",
                    "sd-server",
                    "--parallel",
                    str(cfg.effective_build_jobs),
                ],
                cwd=src,
            ),
            InstallStep("Installing sd-server", action=lambda emit: _install_source(cfg, emit)),
        ]
    )
    return steps


def _extract_release(archive: Path, destination: Path, emit: shared.Emit) -> None:
    """Materialize internal ZIP library symlinks; reject paths outside staging.

    Upstream Unix ZIPs use `zip -y` for .so/.dylib links. Copying their targets
    also works on Windows without symlink privileges.
    """
    with zipfile.ZipFile(archive) as bundle:
        links = []
        for member in bundle.infolist():
            target = shared._safe_archive_target(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                link = bundle.read(member).decode("utf-8")
                if PurePosixPath(link).is_absolute() or "\\" in link or ":" in link:
                    raise ProvisionError(f"Unsafe library symlink: {member.filename}")
                source = (target.parent / link).resolve()
                if not source.is_relative_to(destination.resolve()):
                    raise ProvisionError(f"Unsafe library symlink: {member.filename}")
                links.append((target, source))
            else:
                bundle.extract(member, destination)
            emit(f"Extracted {member.filename}", False)
        while links:
            remaining = []
            for target, source in links:
                if source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                else:
                    remaining.append((target, source))
            if len(remaining) == len(links):
                raise ProvisionError("Unresolved library symlink in release archive")
            links = remaining


def _write_state(cfg: shared.ProvisionConfig, version: str, method: str) -> None:
    state = {
        "engine": "stable-diffusion.cpp",
        "version": version,
        "requested_version": cfg.version,
        "method": method,
        "accelerator": cfg.accelerator,
        "platform": cfg.platform,
        "arch": cfg.arch,
        "executable": str(cfg.provision_dir / "bin" / cfg.exe_name),
        "cuda_architectures": shared.cuda_cmake_architectures(cfg.cuda_arch),
        "cmake_args": cfg.cmake_args,
        "installed_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = cfg.provision_dir / "active-runtime.json"
    temporary = path.with_suffix(".json.next")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_active_runtime(root: Path) -> dict[str, Any]:
    state = shared.read_active_runtime(root)
    return state if state.get("engine") == "stable-diffusion.cpp" else {}


def _stage_install(
    cfg: shared.ProvisionConfig, root: Path, version: str, method: str, emit: shared.Emit
) -> None:
    server = shared._find_server(root, cfg.exe_name)
    if server is None:
        raise ProvisionError(f"{cfg.exe_name} not found under {root}")
    next_bin = cfg.provision_dir / "bin.next"
    shutil.rmtree(next_bin, ignore_errors=True)
    shutil.copytree(server.parent, next_bin, symlinks=False)
    for pattern in ("*.dll", "*.so", "*.so.*", "*.dylib"):
        for library in root.rglob(pattern):
            if library.is_file():
                shutil.copy2(library, next_bin / library.name)
    binary = next_bin / cfg.exe_name
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    emit(shared.validate_server(binary), False)
    shared._swap_staged_bin(cfg.provision_dir / "bin", next_bin)
    _write_state(cfg, version, method)


def _install_source(cfg: shared.ProvisionConfig, emit: shared.Emit) -> None:
    version = shared._run_text(
        ["git", "-C", str(cfg.source_dir), "describe", "--tags", "--always"]
    ).splitlines()
    _stage_install(
        cfg, cfg.build_dir / "bin", version[0] if version else cfg.version, "source", emit
    )


def build_install_plan(cfg: shared.ProvisionConfig) -> list[InstallStep]:
    if cfg.install_method == "source":
        return build_source_plan(cfg)

    def install(emit: shared.Emit) -> None:
        driver = shared.detect_driver_cuda_version() if cfg.accelerator == "cuda" else None
        for release in fetch_releases(cfg.version):
            assets = {
                item["name"]: item["browser_download_url"] for item in release.get("assets", [])
            }
            try:
                names = select_release_assets(list(assets), cfg, driver)
            except ProvisionError:
                continue
            staging = cfg.release_dir / "staging"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            for name in names:
                # GitHub names are external input, including the download destination.
                if Path(name).name != name:
                    raise ProvisionError(f"Unsafe release asset name: {name}")
                archive = shared._safe_archive_target(cfg.release_dir, name)
                emit(f"Downloading {name}", False)
                shared._download(assets[name], archive, emit)
                _extract_release(archive, staging, emit)
            _stage_install(cfg, staging, str(release["tag_name"]), "release", emit)
            return
        raise ProvisionError("No compatible stable-diffusion.cpp release found")

    return [InstallStep("Installing matched stable-diffusion.cpp release", action=install)]


@dataclass
class UpdateStatus:
    installed_version: str
    latest_version: str = ""
    update_available: bool = False
    action: str = "none"
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def check_for_update(cfg: shared.ProvisionConfig, installed_version: str) -> UpdateStatus:
    status = UpdateStatus(installed_version)
    try:
        releases = fetch_releases()
        newest = next(r for r in releases if r.get("assets") and r.get("tag_name"))
        status.latest_version = str(newest["tag_name"])
        # Tags contain commit count plus hash; never compare llama's bNNNN numbering.
        installed = re.fullmatch(r"master-(\d+)-[0-9a-f]+(?:-(\d+)-g[0-9a-f]+)?", installed_version)
        latest = re.fullmatch(r"master-(\d+)-[0-9a-f]+", status.latest_version)
        current = installed_version == status.latest_version or bool(
            installed
            and latest
            and int(installed.group(1)) + int(installed.group(2) or 0) > int(latest.group(1))
        )
        status.update_available = not current
        if not current:
            try:
                select_release_assets(
                    [a["name"] for a in newest["assets"]],
                    cfg,
                    shared.detect_driver_cuda_version() if cfg.accelerator == "cuda" else None,
                )
                status.action = "compile" if cfg.install_method == "source" else "release"
            except ProvisionError:
                status.action = "unavailable" if cfg.install_method == "release" else "compile"
    except Exception as exc:
        status.error = str(exc) or "No stable-diffusion.cpp releases found"
    return status


class SdCppProvisioner(shared.LlamaProvisioner):
    """Use the common runner with sd.cpp's acquisition and build definitions."""

    engine_name = "stable-diffusion.cpp"
    log_prefix = "sdcpp"

    def _build_install_plan(self, cfg: shared.ProvisionConfig) -> list[InstallStep]:
        return build_install_plan(cfg)

    def _release_asset_patterns(self, cfg: shared.ProvisionConfig) -> list[str]:
        return release_asset_patterns(cfg)

    def _ensure_runtime(self) -> Path:
        state = shared.read_active_runtime(self.config.provision_dir)
        if state and state.get("engine") != self.engine_name:
            raise ProvisionError(
                "Cache directory belongs to a different engine; choose another --cache-dir"
            )
        return super()._ensure_runtime()
