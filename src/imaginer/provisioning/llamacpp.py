"""Managed llama.cpp acquisition, adapted from Batch-Image-Interrogator.

See LICENSE.Batch-Image-Interrogator for the original MIT notice.

Resolves an existing llama-server, otherwise installs an environment-matched
official GitHub release, and falls back to a source build for the combinations
that have no compatible published binary.

Ported from Mantic-Mind's `src/node/llama_cpp_provisioner.cpp`, which carries
the hardware-specific knowledge this module depends on. The plan builders are
pure: they spawn no processes, so the accelerator matrix and the CUDA
architecture mapping are unit-testable without a toolchain.
"""

from __future__ import annotations

import json
import os
import platform as platform_module
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from queue import Empty, Queue
from threading import Thread
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from imaginer.provisioning.log_retention import RetentionPolicy, prune_logs, truncate_log

REPO_URL = "https://github.com/ggml-org/llama.cpp"
API_BASE = "https://api.github.com/repos/ggml-org/llama.cpp"
USER_AGENT = "stable-imaginer-llama-provisioner/1.0"

INSTALL_METHODS = ("auto", "release", "source")

# Backends the source builder knows how to configure. Release availability is a
# separate question answered by `release_accelerators()`.
ACCELERATORS = (
    "cuda",
    "rocm",
    "vulkan",
    "metal",
    "sycl-fp32",
    "sycl-fp16",
    "openvino",
    "cpu",
)

# Per-backend source-build prerequisites beyond git/cmake/C++ compiler.
BACKEND_TOOLS: dict[str, tuple[str, str]] = {
    "cuda": ("nvcc", "CUDA Toolkit"),
    "rocm": ("hipcc", "ROCm/HIP SDK"),
    "vulkan": ("glslc", "Vulkan SDK"),
    "sycl-fp32": ("icpx", "Intel oneAPI DPC++ compiler"),
    "sycl-fp16": ("icpx", "Intel oneAPI DPC++ compiler"),
}

PROGRESS_RE = re.compile(r"\[\s*(\d+)\s*%\]")

BUILD_LOG_GLOB = "llama-provision-*.log"

# Build logs share a directory with the far larger server logs but keep their
# own budget, so one long inference run cannot evict the record of a failed
# install. Sized generously in file count and modestly in bytes: a failed build
# is diagnosed from the log the dialog names, and several attempts often sit
# between noticing a problem and fixing it.
BUILD_LOG_RETENTION = RetentionPolicy(
    keep_files=10,
    keep_total_bytes=32 * 1024 * 1024,
    max_bytes=8 * 1024 * 1024,
    tail_bytes=1024 * 1024,
)


class ProvisionError(RuntimeError):
    """Raised when llama.cpp provisioning cannot complete."""


class ProvisionCancelled(ProvisionError):
    """Raised when the caller cancelled an in-flight provisioning run."""


@contextmanager
def _install_lock(root: Path) -> Iterator[None]:
    """An OS lock is released even if the installer crashes; keep the inode in place."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "install.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProvisionError(f"Another installation is using {root}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProvisionConfig:
    """Inputs for one provisioning attempt."""

    provision_dir: Path = Path("cache/llama_cpp")
    install_method: str = "auto"  # auto|release|source
    version: str = "latest"  # git ref / release tag, or "latest"
    # cuda|rocm|vulkan|metal|sycl-*|openvino|cpu. Empty means "detect".
    accelerator: str = ""
    # CUDA compute capability for the source build, e.g. "121" for the DGX Spark
    # GB10. Empty means detect from nvidia-smi.
    cuda_arch: str = ""
    # Extra -D flags appended verbatim to the CMake configure step.
    cmake_args: list[str] = field(default_factory=list)
    # Concurrent compiler jobs. 0 selects a conservative managed default: CUDA
    # and ROCm translation units are memory-hungry enough to OOM a parallel
    # build on modest machines.
    build_jobs: int = 0
    # Skips the preflight probes for one attempt. The troubleshooting path sets
    # this when the user chooses "compile anyway".
    bypass_environment_checks: bool = False
    # Reinstall even when an active runtime already matches the requested
    # accelerator. Used by the CLI's --force option.
    force_reinstall: bool = False
    # cuda-12|cuda-13|... Empty means auto-select from the release asset matrix.
    release_variant: str = ""
    # Test hooks; empty means use the build host.
    platform: str = ""  # windows|linux|macos
    arch: str = ""  # x64|arm64

    def normalized(self) -> ProvisionConfig:
        """Return a copy with defaults resolved against the current host."""
        cfg = ProvisionConfig(
            provision_dir=Path(self.provision_dir).expanduser().resolve(),
            install_method=normalize_install_method(self.install_method),
            version=(self.version or "latest").strip() or "latest",
            accelerator=(self.accelerator or "").strip().lower(),
            cuda_arch=(self.cuda_arch or "").strip(),
            cmake_args=list(self.cmake_args),
            build_jobs=max(0, int(self.build_jobs)),
            bypass_environment_checks=bool(self.bypass_environment_checks),
            force_reinstall=bool(self.force_reinstall),
            release_variant=(self.release_variant or "").strip().lower(),
            platform=(self.platform or "").strip().lower() or current_platform(),
            arch=(self.arch or "").strip().lower() or current_arch(),
        )
        cfg.arch = normalize_arch(cfg.arch)
        if not cfg.accelerator:
            cfg.accelerator = detect_accelerator(cfg.platform)
        if cfg.accelerator not in ACCELERATORS:
            raise ProvisionError(
                f"Unsupported llama.cpp accelerator '{cfg.accelerator}'. "
                f"Choose one of: {', '.join(ACCELERATORS)}"
            )
        if cfg.accelerator == "cuda" and not cfg.cuda_arch:
            cfg.cuda_arch = detect_cuda_arch()
        if cfg.accelerator == "cuda" and cfg.cuda_arch and not cuda_arch_targets(cfg.cuda_arch):
            raise ProvisionError(
                f"Invalid CUDA architecture '{cfg.cuda_arch}'. "
                "Use a compute capability such as 86, 120, or 121."
            )
        if cfg.version.startswith("-"):
            raise ProvisionError("llama.cpp version/ref must not start with '-'")
        return cfg

    @property
    def exe_name(self) -> str:
        return "llama-server.exe" if self.platform == "windows" else "llama-server"

    @property
    def release_dir(self) -> Path:
        return Path(self.provision_dir) / "release"

    @property
    def source_dir(self) -> Path:
        return Path(self.provision_dir) / "llama.cpp-src"

    @property
    def build_dir(self) -> Path:
        """Per-accelerator build tree.

        Reusing a CPU or Vulkan CMake cache for CUDA (or the reverse) produces a
        deceptively successful reconfigure followed by a broken or incorrectly
        linked binary, so each OS/arch/accelerator gets its own cache.
        """
        variant = "-".join(map(_safe_component, (self.platform, self.arch, self.accelerator)))
        return self.source_dir / "build" / variant

    @property
    def effective_build_jobs(self) -> int:
        if self.build_jobs > 0:
            return self.build_jobs
        return 2 if self.accelerator in ("cuda", "rocm") else 4


def normalize_install_method(method: str) -> str:
    value = (method or "").strip().lower()
    return value if value in INSTALL_METHODS else "auto"


def current_platform() -> str:
    system = platform_module.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    return "linux" if system == "linux" else system


def current_arch() -> str:
    return normalize_arch(platform_module.machine())


def normalize_arch(machine: str) -> str:
    value = (machine or "").strip().lower()
    if value in ("x86_64", "amd64", "x64", "win64"):
        return "x64"
    if value in ("arm64", "aarch64"):
        return "arm64"
    return value


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    return cleaned.strip("-") or "default"


# ---------------------------------------------------------------------------
# CUDA architecture handling
# ---------------------------------------------------------------------------


def cuda_arch_targets(value: str) -> list[str]:
    """Parse a CUDA architecture list into bare numeric targets.

    Accepts `sm_121`, `compute_121`, `121-real`, and `;`/`,`/whitespace
    separated lists, returning `["121"]`-style tokens.
    """
    targets: list[str] = []
    for raw in re.split(r"[;,\s]+", value or ""):
        token = raw.strip().lower()
        if not token:
            continue
        for prefix in ("compute_", "sm_"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
                break
        for suffix in ("-real", "-virtual"):
            if token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        if token.endswith("a") and token[:-1].isdigit():
            token = token[:-1]
        if token.isdigit() and token not in targets:
            targets.append(token)
    return targets


def cuda_arch_needs_architecture_specific_target(target: str) -> bool:
    """True for the Blackwell 12X targets that require an `a` suffix."""
    return len(target) == 3 and target.startswith("12") and target[2].isdigit()


def cuda_feature_target(target: str) -> str:
    """The architecture-specific spelling nvcc must support, e.g. 121 -> 121a."""
    return target + "a" if cuda_arch_needs_architecture_specific_target(target) else target


def cuda_cmake_architectures(value: str) -> str:
    """Map detected architectures to the CMAKE_CUDA_ARCHITECTURES value to use.

    llama.cpp's Blackwell MXFP4 kernels use architecture-specific tensor-core
    instructions. Plain `12X` is a baseline/forward-compatible target and cannot
    assemble those instructions, so a detected 12X device must be compiled as
    `12Xa`. `-real` avoids emitting an unnecessary, non-forward-compatible PTX
    payload and keeps CMake 3.28 working under WSL.
    """
    mapped = [
        target + "a-real" if cuda_arch_needs_architecture_specific_target(target) else target
        for target in cuda_arch_targets(value)
    ]
    return ";".join(mapped)


def cuda_arch_requirement(targets: Sequence[str]) -> str:
    """Human-readable toolkit requirement for the troubleshooting report."""
    labels = ", ".join(f"compute_{cuda_feature_target(t)}" for t in targets)
    requirement = f"NVCC support for {labels}"
    if "120" in targets:
        requirement += " (CUDA Toolkit 12.8 or newer; llama.cpp FP4 kernels require sm_120a)"
    return requirement


def nvcc_supports_arch(list_gpu_arch_output: str, target: str) -> bool:
    """Whether `nvcc --list-gpu-arch` output covers a feature target.

    The flag enumerates BASE architectures. CUDA 12 also printed the
    architecture-specific entries (compute_90a, compute_120a); CUDA 13 prints
    none of them, so a 13.0 toolkit that compiles sm_121a happily lists only
    compute_121. Either spelling is therefore accepted.

    This is not a formality: a toolkit too old for the GPU lists neither form
    (CUDA 12.7 stops at compute_90, so sm_120a and sm_120 both miss), which is
    the case this gate exists to catch. Whether the `a` variant itself assembles
    is settled by the probe compile that runs immediately after.
    """
    base = target[:-1] if target.endswith("a") else target
    tokens = set(re.split(r"\s+", list_gpu_arch_output.strip()))
    return f"compute_{target}" in tokens or f"compute_{base}" in tokens


def accelerator_cmake_flags(cfg: ProvisionConfig) -> list[str]:
    """CMake flags that select the accelerator backend."""
    if cfg.accelerator == "cuda":
        flags = ["-DGGML_CUDA=ON"]
        if cfg.cuda_arch:
            # This cache variable is visible before enable_language(CUDA), so it
            # fixes CUDA 13's removed sm_52 compiler-id default without a global
            # -arch flag. A global CMAKE_CUDA_FLAGS=-arch=sm_120 would leak into
            # every target and break llama.cpp's sm_120a MXFP4 compilation even
            # after upstream selects the correct target.
            flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_cmake_architectures(cfg.cuda_arch)}")
        return flags
    if cfg.accelerator == "rocm":
        return ["-DGGML_HIP=ON"]
    if cfg.accelerator == "vulkan":
        return ["-DGGML_VULKAN=ON"]
    if cfg.accelerator == "metal":
        return ["-DGGML_METAL=ON"]
    if cfg.accelerator == "openvino":
        return ["-DGGML_OPENVINO=ON"]
    if cfg.accelerator in ("sycl-fp32", "sycl-fp16"):
        # llama.cpp exposes one SYCL toggle. Precision stays a runtime/device
        # capability; the variants remain distinct compatibility choices without
        # inventing a CMake flag that does not exist.
        return ["-DGGML_SYCL=ON"]
    return []  # cpu: default backend, no flag


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def _run_text(argv: Sequence[str], timeout: float = 10.0) -> str:
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            timeout=timeout,
            check=False,
            text=True,
        )
    except Exception:
        return ""
    return f"{completed.stdout}\n{completed.stderr}".strip()


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def detect_accelerator(platform_id: str = "") -> str:
    """Best accelerator for this host, by probing the installed toolchains."""
    platform_id = platform_id or current_platform()
    if platform_id == "macos":
        return "metal"
    if has_tool("nvcc") or has_tool("nvidia-smi"):
        return "cuda"
    if has_tool("hipcc") or Path("/opt/rocm").exists():
        return "rocm"
    if has_tool("glslc") or has_tool("vulkaninfo"):
        return "vulkan"
    return "cpu"


def detect_cuda_arch() -> str:
    """Detected compute capability as a bare target, e.g. "121"."""
    output = _run_text(("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"))
    match = re.search(r"(\d+)\.(\d+)", output)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return ""


def detect_driver_cuda_version() -> tuple[int, ...] | None:
    """CUDA version reported by the driver, for release-asset selection.

    This is deliberately driver-side. A prebuilt release only needs a driver new
    enough to load it; a SOURCE build needs an actual toolkit, which is what the
    nvcc probes check. Conflating the two is the most common way to end up
    either rejecting a usable release or attempting an impossible compile.
    """
    output = _run_text(("nvidia-smi",))
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(errors="ignore").lower()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Release asset matching
# ---------------------------------------------------------------------------


def release_asset_patterns(cfg: ProvisionConfig) -> list[str]:
    """Regexes matching the release assets needed for this configuration.

    Windows ships a CPU base archive that backend archives layer on top of, so
    a Windows CUDA install is base + cuda + cudart rather than one download.
    Linux and macOS archives are self-contained.
    """
    arch = re.escape(cfg.arch)
    backend = cfg.accelerator
    if cfg.platform == "linux":
        table = {
            "cpu": [rf"^llama-.*-bin-ubuntu-{arch}\.tar\.gz$"],
            "vulkan": [rf"^llama-.*-bin-ubuntu-vulkan-{arch}\.tar\.gz$"],
            "rocm": [rf"^llama-.*-bin-ubuntu-rocm-.*-{arch}\.tar\.gz$"],
            "cuda": [rf"^llama-.*-bin-ubuntu-cuda-[0-9.]+-{arch}\.tar\.gz$"],
            "openvino": [rf"^llama-.*-bin-ubuntu-openvino-.*-{arch}\.tar\.gz$"],
            "sycl-fp32": [rf"^llama-.*-bin-ubuntu-sycl-fp32-{arch}\.tar\.gz$"],
            "sycl-fp16": [rf"^llama-.*-bin-ubuntu-sycl-fp16-{arch}\.tar\.gz$"],
        }
        return table.get(backend, [])
    if cfg.platform == "macos":
        return [rf"^llama-.*-bin-macos-{arch}\.tar\.gz$"] if backend in ("metal", "cpu") else []
    if cfg.platform == "windows":
        base = rf"^llama-.*-bin-win-cpu-{arch}\.zip$"
        table = {
            "cpu": [base],
            "cuda": [base, rf"^llama-.*-bin-win-cuda-([0-9.]+)-{arch}\.zip$"],
            "vulkan": [base, rf"^llama-.*-bin-win-vulkan-{arch}\.zip$"],
            "rocm": [base, rf"^llama-.*-bin-win-(?:hip|rocm).*-{arch}\.zip$"],
            "openvino": [rf"^llama-.*-bin-win-openvino-.*-{arch}\.zip$"],
            "sycl-fp32": [rf"^llama-.*-bin-win-sycl-{arch}\.zip$"],
            "sycl-fp16": [rf"^llama-.*-bin-win-sycl-{arch}\.zip$"],
        }
        return table.get(backend, [])
    return []


def choose_cuda_release(names: Sequence[str], driver_version: tuple[int, ...] | None) -> str | None:
    """Highest CUDA release asset the installed driver can load."""
    versioned: list[tuple[tuple[int, ...], str]] = []
    for name in names:
        match = re.search(r"-cuda-([0-9.]+)-", name)
        if not match:
            continue
        try:
            versioned.append((tuple(int(p) for p in match.group(1).split(".")), name))
        except ValueError:
            continue
    if not versioned:
        return None
    if driver_version is None:
        # Unknown driver: the lowest CUDA build is the safest bet.
        return min(versioned, key=lambda item: item[0])[1]
    eligible = [item for item in versioned if item[0] <= driver_version]
    if not eligible:
        return None
    return max(eligible, key=lambda item: item[0])[1]


def select_release_assets(
    asset_names: Sequence[str],
    cfg: ProvisionConfig,
    driver_version: tuple[int, ...] | None = None,
) -> list[str]:
    """Asset names to download for this configuration, in install order."""
    patterns = release_asset_patterns(cfg)
    if not patterns:
        raise ProvisionError(
            f"No official llama.cpp release covers {cfg.platform}/{cfg.arch}/{cfg.accelerator}"
        )

    selected: list[str] = []
    for pattern in patterns:
        matches = [name for name in asset_names if re.match(pattern, name)]
        if cfg.accelerator == "cuda" and "-cuda-" in pattern:
            targets = [int(t) for t in cuda_arch_targets(cfg.cuda_arch)]
            minimum = (13, 0) if any(t >= 121 for t in targets) else (12, 8)
            if any(t >= 120 for t in targets):
                compatible = []
                for name in matches:
                    version = re.search(r"-cuda-([0-9.]+)-", name)
                    if version and tuple(map(int, version.group(1).split("."))) >= minimum:
                        compatible.append(name)
                matches = compatible
            if cfg.release_variant in ("cuda-12", "cuda-13"):
                major = cfg.release_variant.split("-")[1]
                matches = [n for n in matches if re.search(rf"-cuda-{major}\.", n)]
            choice = choose_cuda_release(matches, driver_version)
            matches = [choice] if choice else []
        if not matches:
            raise ProvisionError(
                f"No llama.cpp release asset matches {pattern} "
                f"for {cfg.platform}/{cfg.arch}/{cfg.accelerator}"
            )
        selected.append(matches[0])

    # Windows CUDA runtime DLLs ship in a companion archive keyed to the chosen
    # CUDA version. Without it the server starts and then fails to load cuBLAS.
    if cfg.platform == "windows" and cfg.accelerator == "cuda":
        cuda_asset = next((n for n in selected if "-cuda-" in n), "")
        match = re.search(r"-cuda-([0-9.]+)-", cuda_asset)
        if match:
            cudart = f"cudart-llama-bin-win-cuda-{match.group(1)}-{cfg.arch}.zip"
            if cudart not in asset_names:
                raise ProvisionError(
                    f"The Windows CUDA release is missing its required runtime archive: {cudart}"
                )
            selected.append(cudart)
    return selected


def release_accelerators(asset_names: Sequence[str], cfg: ProvisionConfig) -> list[str]:
    """Accelerators with an official release for this OS/arch, UI display order."""
    available: list[str] = []
    for backend in (
        "cuda",
        "rocm",
        "vulkan",
        "metal",
        "sycl-fp32",
        "sycl-fp16",
        "openvino",
        "cpu",
    ):
        probe = ProvisionConfig(
            provision_dir=cfg.provision_dir,
            accelerator=backend,
            platform=cfg.platform,
            arch=cfg.arch,
        )
        patterns = release_asset_patterns(probe)
        if not patterns:
            continue
        if all(any(re.match(p, name) for name in asset_names) for p in patterns):
            available.append(backend)
    return available


def _fetch_json(suffix: str, timeout: float) -> Any:
    request = Request(
        f"{API_BASE}/{suffix}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_releases(version: str = "latest", timeout: float = 30.0) -> list[dict[str, Any]]:
    """Fetch release candidates from the llama.cpp GitHub API.

    Upstream's stable ``latest`` release can be metadata-only and link to a
    nightly ``bNNNN`` release that owns the platform archives. Querying the
    releases collection lets the caller select the newest candidate that
    actually contains the requested runtime instead of falling straight into a
    source build on every machine.
    """
    if version != "latest":
        payload = _fetch_json(f"releases/tags/{quote(version, safe='')}", timeout)
        if not isinstance(payload, dict):
            raise ProvisionError(f"Invalid release metadata returned for {version}")
        return [payload]

    payload = _fetch_json("releases?per_page=20", timeout)
    if not isinstance(payload, list):
        raise ProvisionError("Invalid release list returned by GitHub")
    return [release for release in payload if isinstance(release, dict)]


def fetch_release(version: str = "latest", timeout: float = 30.0) -> dict[str, Any]:
    """Backward-compatible single-release view of :func:`fetch_releases`."""
    releases = fetch_releases(version, timeout)
    if not releases:
        raise ProvisionError("GitHub returned no llama.cpp releases")
    return releases[0]


def select_release_candidate(
    releases: Sequence[dict[str, Any]],
    cfg: ProvisionConfig,
    driver_version: tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """First release candidate containing every archive required by ``cfg``."""
    errors: list[str] = []
    for release in releases:
        tag = str(release.get("tag_name", "")).strip() or "unknown"
        names = [str(asset.get("name", "")) for asset in release.get("assets") or []]
        try:
            selected = select_release_assets(names, cfg, driver_version)
        except ProvisionError as exc:
            errors.append(f"{tag}: {exc}")
            continue
        return release, selected

    error_summary = "\n".join(errors[:5])
    detail = f"\n{error_summary}" if error_summary else ""
    raise ProvisionError(
        f"No recent llama.cpp release contains a complete runtime for "
        f"{cfg.platform}/{cfg.arch}/{cfg.accelerator}.{detail}"
    )


# ---------------------------------------------------------------------------
# Update checking
# ---------------------------------------------------------------------------

BUILD_TAG_RE = re.compile(r"\bb(\d+)\b", re.IGNORECASE)

# How far behind upstream a runtime must fall before an update is worth
# reporting. llama.cpp tags roughly a dozen nightly builds a day, so any
# smaller number means a source install tracking master reports an update
# permanently — and the signal stops meaning anything. 100 builds is about a
# week of upstream activity at that cadence, which is a reasonable floor for
# spending twenty minutes on a recompile.
UPDATE_BUILD_THRESHOLD = 100


def release_build_number(text: str) -> int | None:
    """The `bNNNN` build number in a tag or version string, if there is one."""
    match = BUILD_TAG_RE.search(text or "")
    return int(match.group(1)) if match else None


def builds_behind(installed: str, latest_tag: str) -> int | None:
    """How many upstream builds separate an installed runtime from `latest_tag`.

    0 means current or ahead. None means the comparison could not be made —
    a custom build string with no recognisable build number — in which case
    callers should report the update rather than silently hide it.
    """
    if not installed or not latest_tag:
        return None
    if latest_tag in installed:
        return 0

    latest_build = release_build_number(latest_tag)
    installed_build = release_build_number(installed)
    if latest_build is None or installed_build is None:
        return None
    return max(0, latest_build - installed_build)


def installed_version_is_current(installed: str, latest_tag: str) -> bool:
    """Whether an installed runtime is already at least as new as `latest_tag`.

    Upstream tags are `bNNNN`, and the recorded version varies by install
    method: a release stores the tag, while a source build stores a
    `git describe` such as `b10828-2-g465e49b9c`. Comparing build numbers
    rather than strings matters for the source case, where a build from master
    is routinely *ahead* of the newest tagged release — string inequality alone
    would report an update forever.
    """
    if not installed or not latest_tag:
        return False

    latest_build = release_build_number(latest_tag)
    installed_build = release_build_number(installed)
    if latest_build is not None and installed_build is not None:
        return installed_build >= latest_build
    return installed.strip() == latest_tag.strip()


@dataclass
class UpdateStatus:
    """Result of an update check."""

    current_version: str = ""
    latest_version: str = ""
    # Past the reporting threshold. A runtime can be behind without this being
    # True; `behind` carries the raw distance either way.
    update_available: bool = False
    # Builds between the installed runtime and upstream. None when the versions
    # could not be compared, 0 when current or ahead.
    behind: int | None = None
    threshold: int = UPDATE_BUILD_THRESHOLD
    # release: a matched asset exists. compile: a source rebuild is required.
    # unavailable: no asset, and the configured method forbids compiling.
    action: str = ""
    warning: str = ""
    release_alternatives: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def is_current(self) -> bool:
        """Level with upstream, or ahead of it."""
        return self.behind == 0

    @property
    def below_threshold(self) -> bool:
        """Behind, but not far enough to be worth acting on."""
        return bool(self.behind) and not self.update_available


def check_for_update(
    cfg: ProvisionConfig,
    installed_version: str,
    timeout: float = 30.0,
    threshold: int = UPDATE_BUILD_THRESHOLD,
) -> UpdateStatus:
    """Ask upstream whether a newer llama.cpp runtime exists for this host.

    Reports not just *that* an update exists but what taking it would cost:
    downloading a release is a minute, recompiling is twenty. The two are worth
    distinguishing before the user commits to one.

    `threshold` suppresses the report until the runtime is meaningfully behind;
    pass 0 to report any difference at all.
    """
    cfg = cfg.normalized()
    status = UpdateStatus(current_version=installed_version, threshold=threshold)

    try:
        releases = fetch_releases("latest", timeout)
    except Exception as exc:
        status.error = f"Could not reach the llama.cpp release API: {exc}"
        return status
    if not releases:
        status.error = "GitHub returned no llama.cpp releases"
        return status

    # The API collection currently puts a metadata-only stable release ahead
    # of the nightly bNNNN builds that contain llama-server. Update currency is
    # defined by those build tags, just like provisioning, so skip unrelated
    # release entries rather than comparing a bNNNN install with e.g. v0.4.0.
    newest = next(
        (
            release
            for release in releases
            if release_build_number(str(release.get("tag_name", ""))) is not None
        ),
        None,
    )
    if newest is None:
        status.error = "GitHub returned no tagged llama.cpp runtime builds"
        return status
    status.latest_version = str(newest.get("tag_name", "")).strip()

    status.behind = builds_behind(installed_version, status.latest_version)
    if status.behind == 0:
        return status
    # An uncomparable version (None) is reported rather than hidden: failing to
    # parse a custom build string is not evidence that it is current.
    if status.behind is not None and status.behind < threshold:
        return status

    status.update_available = True
    asset_names = [str(asset.get("name", "")) for asset in newest.get("assets") or []]
    available = release_accelerators(asset_names, cfg)
    status.release_alternatives = [a for a in available if a != cfg.accelerator]

    try:
        select_release_assets(
            asset_names,
            cfg,
            detect_driver_cuda_version() if cfg.accelerator == "cuda" else None,
        )
        matched_release = True
    except ProvisionError:
        # A backend name in the asset list is not sufficient on its own. CUDA
        # releases must also fit the installed driver (and Windows builds need
        # their matching cudart archive), exactly as provisioning requires.
        matched_release = False

    release_present = cfg.accelerator in available
    if matched_release and cfg.install_method != "source":
        status.action = "release"
    elif cfg.install_method == "release":
        status.action = "unavailable"
        reason = "No compatible official" if release_present else "No official"
        status.warning = (
            f"{reason} {cfg.platform}/{cfg.arch}/{cfg.accelerator} asset exists for "
            f"{status.latest_version}, and the install method is restricted to releases."
        )
    else:
        status.action = "compile"
        if cfg.install_method == "source":
            reason = "This runtime was installed from source."
        elif release_present:
            reason = (
                f"No official {cfg.platform}/{cfg.arch}/{cfg.accelerator} asset for "
                f"{status.latest_version} is compatible with this system."
            )
        else:
            reason = (
                f"No official {cfg.platform}/{cfg.arch}/{cfg.accelerator} asset exists for "
                f"{status.latest_version}."
            )
        status.warning = (
            f"{reason} Updating will compile llama-server from source, "
            "which can take several minutes."
        )

    if status.release_alternatives and not release_present:
        status.warning += (
            f" Official releases do exist for: {', '.join(status.release_alternatives)}."
        )
    return status


# ---------------------------------------------------------------------------
# Install steps
# ---------------------------------------------------------------------------

Emit = Callable[[str, bool], None]
"""Sink for a live output line: (text, is_stderr)."""


@dataclass
class InstallStep:
    """One ordered step in an install plan.

    Either `argv` (a subprocess) or `action` (an in-process callable). Native
    actions cover download/extract/install, where shelling out would only add a
    dependency on an external interpreter and a layer of quoting.
    """

    label: str
    argv: list[str] | None = None
    action: Callable[[Emit], None] | None = None
    cwd: Path | None = None
    allow_failure: bool = False


def build_release_plan(cfg: ProvisionConfig) -> list[InstallStep]:
    """Download and install a matched official release."""

    def install_release(emit: Emit) -> None:
        releases = fetch_releases(cfg.version)
        driver_version = detect_driver_cuda_version() if cfg.accelerator == "cuda" else None
        release, names = select_release_candidate(releases, cfg, driver_version)
        tag = str(release.get("tag_name", "")).strip()
        assets = {
            str(a.get("name", "")): str(a.get("browser_download_url", ""))
            for a in release.get("assets") or []
        }
        emit(f"Release {tag or cfg.version}: selected {', '.join(names)}", False)

        release_root = cfg.release_dir
        staging = release_root / "staging"
        next_bin = release_root / "bin.next"
        # Both install methods land here, so the setup scripts and any saved
        # Inquiry path keep pointing at one stable location.
        bin_dir = Path(cfg.provision_dir) / "bin"
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(next_bin, ignore_errors=True)
        staging.mkdir(parents=True)
        next_bin.mkdir(parents=True)

        for index, name in enumerate(names, 1):
            archive = release_root / "downloads" / name
            archive.parent.mkdir(parents=True, exist_ok=True)
            emit(f"[{index}/{len(names)}] Downloading {name}", False)
            _download(assets[name], archive, emit)
            _extract(archive, staging, emit)
            emit(f"Extracted {name}", False)

        server = _find_server(staging, cfg.exe_name)
        if server is None:
            raise ProvisionError("llama-server not found in the selected release assets")

        # Copy the whole directory so companion shared libraries travel with the
        # executable, then layer in libraries that landed elsewhere in staging
        # (Windows backend archives unpack alongside, not into, the base).
        shutil.copytree(server.parent, next_bin, dirs_exist_ok=True, symlinks=True)
        for pattern in ("*.dll", "*.so", "*.so.*", "*.dylib"):
            for library in staging.rglob(pattern):
                if library.is_file():
                    shutil.copy2(library, next_bin / library.name)

        target = next_bin / cfg.exe_name
        if not target.exists():
            shutil.copy2(server, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        emit(validate_server(target), False)

        _swap_staged_bin(bin_dir, next_bin)
        shutil.rmtree(staging, ignore_errors=True)
        _write_version(cfg, bin_dir / cfg.exe_name, tag or cfg.version, "release")
        emit(f"Installed llama-server from release {tag or cfg.version}", False)

    return [
        InstallStep(
            "Downloading matched llama.cpp release", action=install_release, cwd=cfg.provision_dir
        )
    ]


def build_source_plan(cfg: ProvisionConfig) -> list[InstallStep]:
    """Clone, configure, and compile llama-server for this accelerator."""
    src = cfg.source_dir
    build = cfg.build_dir
    ref = "master" if cfg.version == "latest" else cfg.version
    plan: list[InstallStep] = []

    if not cfg.bypass_environment_checks:
        plan.append(
            InstallStep(
                "Checking source-build prerequisites",
                action=lambda emit: preflight(cfg, emit),
                cwd=cfg.provision_dir,
            )
        )

    if not (src / ".git").exists():
        plan.append(
            InstallStep(
                "Cloning llama.cpp source",
                argv=["git", "clone", REPO_URL, str(src)],
                cwd=cfg.provision_dir,
            )
        )
    plan.append(
        InstallStep("Fetching tags", argv=["git", "fetch", "--tags", "--all", "--prune"], cwd=src)
    )
    plan.append(InstallStep(f"Checking out {ref}", argv=["git", "checkout", ref], cwd=src))
    if cfg.version == "latest":
        plan.append(InstallStep("Updating source", argv=["git", "pull", "--ff-only"], cwd=src))

    if cfg.accelerator == "cuda":
        # CMake persists CMAKE_CUDA_COMPILER across attempts. That is especially
        # hazardous after upgrading from a distro /usr/bin/nvcc to a newer
        # toolkit selected through CUDACXX/PATH: the old compiler keeps winning
        # even though every live probe sees the new one. Clear only the generated
        # configure state; the checkout stays intact.
        plan.append(
            InstallStep(
                "Clearing cached CUDA compiler selection",
                argv=["cmake", "-E", "remove_directory", str(build / "CMakeFiles")],
                cwd=src,
                allow_failure=True,
            )
        )
        plan.append(
            InstallStep(
                "Clearing cached CUDA configure state",
                argv=["cmake", "-E", "remove", str(build / "CMakeCache.txt")],
                cwd=src,
                allow_failure=True,
            )
        )

    configure = [
        "cmake",
        "-S",
        str(src),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
        *accelerator_cmake_flags(cfg),
        *cfg.cmake_args,
    ]
    if cfg.platform != "windows":
        # Resolve the ggml/llama shared libraries next to the executable rather
        # than through an absolute path into the build tree. Without this the
        # installed binary keeps a RUNPATH into a directory the user is
        # otherwise free to delete to reclaim several GB, and it only works
        # today because the runtime manager happens to set LD_LIBRARY_PATH.
        configure.append("-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON")
        configure.append(
            "-DCMAKE_INSTALL_RPATH=" + ("@loader_path" if cfg.platform == "macos" else "$ORIGIN")
        )
    plan.append(InstallStep("Configuring llama.cpp (CMake)", argv=configure, cwd=src))
    plan.append(
        InstallStep(
            "Building llama-server",
            argv=[
                "cmake",
                "--build",
                str(build),
                "--config",
                "Release",
                "--target",
                "llama-server",
                "--parallel",
                str(cfg.effective_build_jobs),
            ],
            cwd=src,
        )
    )
    plan.append(
        InstallStep(
            "Installing built llama-server",
            action=lambda emit: _install_source_build(cfg, emit),
            cwd=src,
        )
    )
    return plan


def build_install_plan(cfg: ProvisionConfig) -> list[InstallStep]:
    """Steps for the configured install method (one attempt, not the ladder)."""
    cfg = cfg.normalized()
    if cfg.install_method == "source":
        return build_source_plan(cfg)
    return build_release_plan(cfg)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def missing_prerequisites(cfg: ProvisionConfig) -> list[str]:
    """Source-build tools that are not on PATH."""
    missing: list[str] = []
    for tool in ("git", "cmake"):
        if not has_tool(tool):
            missing.append(tool)

    if cfg.platform != "windows":
        compiler = os.environ.get("CXX", "c++")
        if not has_tool(compiler):
            missing.append(f"{compiler} (C++ compiler)")

    if cfg.accelerator == "cuda":
        nvcc = os.environ.get("CUDACXX", "nvcc")
        if not has_tool(nvcc) and not Path(nvcc).is_file():
            missing.append("nvcc (CUDA Toolkit; the display driver alone is insufficient)")
    elif cfg.accelerator == "openvino":
        if not (os.environ.get("OpenVINO_DIR") or os.environ.get("INTEL_OPENVINO_DIR")):  # noqa: SIM112
            missing.append("OpenVINO toolkit environment")
    else:
        entry = BACKEND_TOOLS.get(cfg.accelerator)
        if entry and not has_tool(entry[0]):
            missing.append(f"{entry[0]} ({entry[1]})")
    return missing


def preflight(cfg: ProvisionConfig, emit: Emit) -> None:
    """Fail fast on a toolchain that cannot produce the requested build.

    Everything here is cheap. Discovering a missing nvcc twenty minutes into a
    compile is the failure mode this exists to prevent.
    """
    missing = missing_prerequisites(cfg)
    if missing:
        detail = ", ".join(missing)
        hint = ""
        if cfg.accelerator == "cuda":
            hint = (
                "\nGPU visibility via nvidia-smi is not enough: install the CUDA Toolkit "
                "in this environment so nvcc is on PATH."
            )
        raise ProvisionError(f"Missing llama.cpp source-build prerequisites: {detail}{hint}")

    emit(_first_line(_run_text(("cmake", "--version"))), False)
    if cfg.platform != "windows":
        emit(_first_line(_run_text((os.environ.get("CXX", "c++"), "--version"))), False)

    if cfg.accelerator == "cuda":
        _preflight_cuda(cfg, emit)

    free_bytes = shutil.disk_usage(cfg.provision_dir).free
    if free_bytes < 4 * 1024**3:
        emit(
            "Warning: less than 4 GiB free for the llama.cpp source build "
            f"({free_bytes // 1024**3} GiB)",
            True,
        )


def _preflight_cuda(cfg: ProvisionConfig, emit: Emit) -> None:
    """Verify nvcc can target every requested architecture, then prove it."""
    nvcc = os.environ.get("CUDACXX", "nvcc")
    emit(_last_line(_run_text((nvcc, "--version"))), False)

    targets = [cuda_feature_target(t) for t in cuda_arch_targets(cfg.cuda_arch)]
    if not targets:
        return

    listed = _run_text((nvcc, "--list-gpu-arch"))
    for target in targets:
        if nvcc_supports_arch(listed, target):
            continue
        hint = ""
        if target == "120a":
            hint = " CUDA Toolkit 12.8 or newer is required for sm_120a."
        elif re.fullmatch(r"12\da", target):
            hint = (
                f" CUDA Toolkit 13.0 or newer is required for sm_{target}; "
                "12.8 covers only sm_120a."
            )
        raise ProvisionError(
            f"Selected nvcc ({shutil.which(nvcc) or nvcc}) "
            f"does not support compute_{target}.{hint} "
            "The CUDA version shown by nvidia-smi is driver compatibility, not the installed "
            "compiler version."
        )

    # `--list-gpu-arch` is a toolkit-version accident; only a compile proves the
    # architecture-specific target actually assembles with this nvcc/ptxas pair.
    probe_arch = targets[0]
    with tempfile.TemporaryDirectory(prefix="llama-cuda-probe-") as tmp:
        probe = Path(tmp) / "probe.cu"
        probe.write_text('extern "C" __global__ void probe() {}\n', encoding="utf-8")
        result = subprocess.run(
            [nvcc, f"-arch=sm_{probe_arch}", "-c", str(probe), "-o", str(Path(tmp) / "probe.o")],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ProvisionError(
                f"CUDA compiler/assembler smoke test failed for sm_{probe_arch}. "
                f"Ensure nvcc and ptxas come from the same CUDA Toolkit installation.\n"
                f"{(result.stdout or '').strip()}"
            )
    emit(f"CUDA probe compile succeeded for sm_{probe_arch}", False)

    if is_wsl():
        emit("WSL detected: using the Linux CUDA toolkit and a conservative parallel build.", False)
        if not has_tool("nvidia-smi"):
            emit(
                "Warning: nvidia-smi is not visible inside WSL; the build can finish, "
                "but CUDA runtime validation may fail.",
                True,
            )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _download(
    url: str,
    destination: Path,
    emit: Emit | None = None,
    timeout: float = 120.0,
) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"})
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as output:
        downloaded = 0
        next_report = 8 * 1024**2
        while True:
            chunk = response.read(1024**2)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if emit is not None and downloaded >= next_report:
                emit(f"Downloaded {downloaded / 1024**2:.0f} MiB", False)
                next_report += 8 * 1024**2


def _safe_archive_target(destination: Path, member_name: str) -> Path:
    """Resolve an archive member while rejecting absolute and traversal paths."""
    posix = PurePosixPath(member_name.replace("\\", "/"))
    windows = PureWindowsPath(member_name)
    if posix.is_absolute() or windows.drive or ".." in posix.parts or ":" in member_name:
        raise ProvisionError(f"Unsafe path in release archive: {member_name}")
    root = destination.resolve()
    target = (destination / member_name).resolve()
    if target != root and root not in target.parents:
        raise ProvisionError(f"Unsafe path in release archive: {member_name}")
    return target


def _extract(archive: Path, destination: Path, emit: Emit | None = None) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            for member in members:
                _safe_archive_target(destination, member.filename)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ProvisionError(f"Symlink in ZIP release archive: {member.filename}")
            for index, member in enumerate(members, 1):
                bundle.extract(member, destination)
                if emit is not None and index % 50 == 0:
                    emit(f"Extracted {index}/{len(members)} files", False)
    elif name.endswith((".tar.gz", ".tgz", ".tar.xz")):
        with tarfile.open(archive, "r:*") as bundle:
            tar_members = bundle.getmembers()
            for tar_member in tar_members:
                _safe_archive_target(destination, tar_member.name)
            for index, tar_member in enumerate(tar_members, 1):
                bundle.extract(tar_member, destination, filter="data")
                if emit is not None and index % 50 == 0:
                    emit(f"Extracted {index}/{len(tar_members)} files", False)
    else:
        raise ProvisionError(f"Unsupported release archive: {archive.name}")


def _find_server(root: Path, exe_name: str) -> Path | None:
    candidates = [p for p in root.rglob(exe_name) if p.is_file()]
    if not candidates:
        return None
    # Prefer the shallowest match: backend archives sometimes ship test copies.
    return min(candidates, key=lambda p: len(p.parts))


def validate_server(binary: Path, timeout: float = 20.0) -> str:
    """Require an installed executable to start successfully."""
    failures: list[str] = []
    for flag in ("--version", "--help"):
        try:
            completed = subprocess.run(
                [str(binary), flag],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"{flag}: {exc}")
            continue
        output = (completed.stdout or "").strip()
        if completed.returncode == 0:
            return _first_line(output) or f"Validated {binary.name} {flag}"
        failures.append(f"{flag}: exit {completed.returncode}: {_last_line(output)}")
    raise ProvisionError(f"Installed {binary.name} failed validation. " + "; ".join(failures))


def _swap_staged_bin(bin_dir: Path, next_bin: Path) -> None:
    """Replace the live runtime while retaining and restoring the previous one."""
    previous = bin_dir.parent / "bin.previous"
    shutil.rmtree(previous, ignore_errors=True)
    bin_dir.parent.mkdir(parents=True, exist_ok=True)
    if bin_dir.exists():
        bin_dir.replace(previous)
    try:
        next_bin.replace(bin_dir)
    except Exception:
        if previous.exists() and not bin_dir.exists():
            previous.replace(bin_dir)
        raise


def _install_source_build(cfg: ProvisionConfig, emit: Emit) -> None:
    """Copy the compiled binary and its libraries into the managed bin dir."""
    built = None
    for candidate in (
        cfg.build_dir / "bin" / "Release" / cfg.exe_name,
        cfg.build_dir / "bin" / cfg.exe_name,
    ):
        if candidate.is_file():
            built = candidate
            break
    if built is None:
        raise ProvisionError(
            f"Build completed but {cfg.exe_name} was not found under {cfg.build_dir}"
        )

    # Stage beside the live directory and swap, so a failed copy cannot leave
    # the caller with no runtime at all. The old binary stays usable until the
    # replacement is complete on disk.
    bin_dir = Path(cfg.provision_dir) / "bin"
    next_bin = Path(cfg.provision_dir) / "bin.next"
    shutil.rmtree(next_bin, ignore_errors=True)
    shutil.copytree(built.parent, next_bin, symlinks=True)
    target = next_bin / cfg.exe_name
    target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    emit(validate_server(target), False)

    _swap_staged_bin(bin_dir, next_bin)
    target = bin_dir / cfg.exe_name

    describe = (
        _run_text(("git", "-C", str(cfg.source_dir), "describe", "--tags", "--always"))
        or cfg.version
    )
    version = f"{_first_line(describe)} (source, {cfg.accelerator}"
    if cfg.accelerator == "cuda" and cfg.cuda_arch:
        version += f", cuda-arch={cuda_cmake_architectures(cfg.cuda_arch)}"
    version += ")"
    _write_version(cfg, target, version, "source")

    emit(_first_line(_run_text((str(target), "--version"))) or f"Installed {target}", False)


def _write_version(cfg: ProvisionConfig, binary: Path, version: str, method: str) -> None:
    record: dict[str, Any] = {
        "version": version,
        "method": method,
        "accelerator": cfg.accelerator,
        "platform": cfg.platform,
        "arch": cfg.arch,
        "executable": str(binary),
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "requested_version": cfg.version,
        "release_variant": cfg.release_variant,
        "cmake_args": cfg.cmake_args,
    }
    if cfg.accelerator == "cuda" and cfg.cuda_arch:
        record["cuda_architectures"] = cuda_cmake_architectures(cfg.cuda_arch)
    state = Path(cfg.provision_dir) / "active-runtime.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Retain the plain-text marker the setup scripts already read.
    (binary.parent / "llama-server.version").write_text(version, encoding="utf-8")


def read_active_runtime(provision_dir: Path) -> dict[str, Any]:
    """Metadata for the currently installed managed runtime, or {}."""
    root = Path(provision_dir)
    try:
        data = json.loads((root / "active-runtime.json").read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass

    # A runtime installed before this state file existed still has the
    # plain-text marker the setup scripts wrote, so report what it says rather
    # than claiming nothing is installed.
    try:
        version = (root / "bin" / "llama-server.version").read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    return {"version": version, "method": "legacy"} if version else {}


def executable_candidates(cfg: ProvisionConfig) -> list[Path]:
    """On-disk locations for a managed llama-server, most specific first."""
    exe = cfg.exe_name
    return [
        Path(cfg.provision_dir) / "bin" / exe,
        cfg.build_dir / "bin" / "Release" / exe,
        cfg.build_dir / "bin" / exe,
    ]


def find_managed_executable(cfg: ProvisionConfig) -> Path | None:
    for candidate in executable_candidates(cfg):
        if candidate.is_file():
            return candidate
    return None


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def _last_line(text: str) -> str:
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def _ladder_rung(
    cfg: ProvisionConfig, ok: bool, detail: str, skipped: bool = False
) -> dict[str, Any]:
    """One rung of the acquisition ladder, as recorded in active-runtime.json."""
    return {
        "method": cfg.install_method,
        "accelerator": cfg.accelerator,
        "ok": bool(ok),
        "skipped": bool(skipped),
        "detail": detail,
    }


def _record_ladder(provision_dir: Path, ladder: list[dict[str, Any]]) -> None:
    """Attach the rungs of the run that installed the runtime to its record.

    Stored beside the version so a front end can show which rung stuck long
    after the provisioning dialog has closed. Skipped when no record exists,
    since the ladder only means something next to an installed runtime.
    """
    state = Path(provision_dir) / "active-runtime.json"
    try:
        record = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(record, dict):
        return
    record["ladder"] = ladder
    state.write_text(json.dumps(record, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class StepProgress:
    """Progress for one running step."""

    step: int
    total: int
    label: str
    fraction: float = -1.0
    last_line: str = ""


ProgressSink = Callable[[StepProgress], None]
CancelCheck = Callable[[], bool]


def parse_progress_fraction(line: str) -> float:
    """CMake's `[ 42%]` build progress as 0..1, or -1 when absent."""
    match = PROGRESS_RE.search(line)
    if not match:
        return -1.0
    return min(1.0, max(0.0, int(match.group(1)) / 100.0))


class LlamaProvisioner:
    """Runs install plans, streaming output and honouring cancellation."""

    engine_name = "llama.cpp"
    log_prefix = "llama"

    def _build_install_plan(self, cfg: ProvisionConfig) -> list[InstallStep]:
        return build_install_plan(cfg)

    def _release_asset_patterns(self, cfg: ProvisionConfig) -> list[str]:
        return release_asset_patterns(cfg)

    def _validate_server(self, binary: Path) -> str:
        return validate_server(binary)

    def __init__(
        self,
        config: ProvisionConfig,
        log_sink: Emit | None = None,
        progress_sink: ProgressSink | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        self.config = config.normalized()
        self.log_sink = log_sink or (lambda line, is_stderr: None)
        self.progress_sink = progress_sink or (lambda progress: None)
        self.cancel_check = cancel_check or (lambda: False)
        self.build_log_path: Path | None = None
        # Set when the ladder had to settle for a weaker backend than requested.
        self.target_mismatch: str = ""
        self.used_existing: bool = False

    def ensure_runtime(self) -> Path:
        """Serialize installations targeting the same cache, including across processes."""
        with _install_lock(self.config.provision_dir):
            self._check_cancel()
            return self._ensure_runtime()

    def _ensure_runtime(self) -> Path:
        """Install llama-server if needed and return its path.

        `auto` tries a matched release first and falls back to a source build,
        for combinations with no compatible published binary.
        """
        existing = find_managed_executable(self.config)
        if (
            existing is not None
            and self.config.install_method == "auto"
            and not self.config.force_reinstall
        ):
            active = read_active_runtime(self.config.provision_dir)
            if (
                active.get("accelerator") == self.config.accelerator
                and active.get("platform") == self.config.platform
                and active.get("arch") == self.config.arch
                and (
                    self.config.version == "latest"
                    or active.get("requested_version") == self.config.version
                    or active.get("version") == self.config.version
                )
                and (
                    not self.config.cuda_arch
                    or active.get("cuda_architectures")
                    == cuda_cmake_architectures(self.config.cuda_arch)
                )
                and (
                    not self.config.release_variant
                    or active.get("release_variant") == self.config.release_variant
                )
                and active.get("cmake_args", []) == self.config.cmake_args
            ):
                try:
                    detail = self._validate_server(existing)
                except ProvisionError as exc:
                    self.log_sink(f"Existing managed runtime is unusable: {exc}", True)
                else:
                    self.used_existing = True
                    self.log_sink(detail, False)
                    self.log_sink(
                        f"Using existing managed {self.config.exe_name}: {existing}", False
                    )
                    return existing

        attempts = self._attempts()
        errors: list[str] = []
        ladder = self._skipped_rungs()
        for index, (label, cfg) in enumerate(attempts, 1):
            self.log_sink(f"=== Attempt {index}/{len(attempts)}: {label} ===", False)
            try:
                self._run_plan(self._build_install_plan(cfg))
            except ProvisionCancelled:
                raise
            except ProvisionError as exc:
                errors.append(f"{label}: {exc}")
                ladder.append(_ladder_rung(cfg, ok=False, detail=_first_line(str(exc))))
                self.log_sink(f"{label} failed: {exc}", True)
                continue

            found = find_managed_executable(cfg)
            if found is None:
                errors.append(f"{label}: completed but produced no {cfg.exe_name}")
                ladder.append(
                    _ladder_rung(cfg, ok=False, detail=f"completed but produced no {cfg.exe_name}")
                )
                continue

            fell_back = cfg.accelerator != self.config.accelerator
            ladder.append(
                _ladder_rung(
                    cfg, ok=True, detail="installed — last rung" if fell_back else "installed"
                )
            )
            _record_ladder(cfg.provision_dir, ladder)

            if fell_back:
                self.target_mismatch = (
                    f"Installed a {cfg.accelerator} runtime, but {self.config.accelerator} "
                    f"was requested. Inference will not use the GPU.\n" + "\n".join(errors)
                )
                self.log_sink(self.target_mismatch, True)
            self.log_sink(f"{cfg.exe_name} ready: {found}", False)
            return found

        raise ProvisionError(f"{self.engine_name} provisioning failed.\n" + "\n".join(errors))

    def _skipped_rungs(self) -> list[dict[str, Any]]:
        """Rungs `_attempts` leaves out, recorded so the ladder reads in full.

        Without this a machine with no published build would show a ladder that
        starts at the source build, hiding why the quick download was never
        tried.
        """
        cfg = self.config
        if cfg.install_method != "auto" or self._release_asset_patterns(cfg):
            return []
        release = replace(cfg, install_method="release")
        return [
            _ladder_rung(
                release,
                ok=False,
                skipped=True,
                detail=f"no published build for {cfg.platform}-{cfg.arch} + {cfg.accelerator}",
            )
        ]

    def _attempts(self) -> list[tuple[str, ProvisionConfig]]:
        return self.install_attempts()

    def install_attempts(self) -> list[tuple[str, ProvisionConfig]]:
        """Ordered attempts for inspection without downloads or compilation."""
        cfg = self.config
        if cfg.install_method != "auto":
            return [(cfg.install_method, cfg)]

        attempts: list[tuple[str, ProvisionConfig]] = []
        if self._release_asset_patterns(cfg):
            attempts.append(("matched release", replace(cfg, install_method="release")))
        attempts.append(("source fallback", replace(cfg, install_method="source")))

        # Last resort. A GPU backend can be impossible on this host in ways
        # neither earlier rung fixes: no published asset AND a toolkit too old
        # to compile one. A working CPU runtime beats no runtime, so long as the
        # downgrade is reported rather than passed off as what was asked for.
        if cfg.accelerator != "cpu":
            cpu = replace(cfg, accelerator="cpu", install_method="release", cuda_arch="")
            if self._release_asset_patterns(cpu):
                attempts.append(("cpu fallback", cpu))
        return attempts

    def _run_plan(self, plan: list[InstallStep]) -> None:
        if not plan:
            raise ProvisionError("install plan is empty")
        self._open_build_log()
        total = len(plan)
        for index, step in enumerate(plan, 1):
            self._check_cancel()
            progress = StepProgress(step=index, total=total, label=step.label)
            self.progress_sink(progress)
            self._log(f"[{self.log_prefix}-install] {step.label}", False)

            try:
                if step.action is not None:

                    def emit_action_line(
                        line: str, is_stderr: bool = False, current: StepProgress = progress
                    ) -> None:
                        self._check_cancel()
                        self._on_line(current, line, is_stderr)

                    step.action(emit_action_line)
                else:
                    self._run_command(step, progress)
                self._check_cancel()
            except ProvisionCancelled:
                raise
            except Exception as exc:
                if step.allow_failure:
                    self._log(f"{step.label} failed (ignored): {exc}", True)
                    continue
                raise ProvisionError(f"step '{step.label}' failed: {exc}") from exc

    def _run_command(self, step: InstallStep, progress: StepProgress) -> None:
        assert step.argv is not None
        cwd = step.cwd or Path(self.config.provision_dir)
        cwd.mkdir(parents=True, exist_ok=True)
        self._log(f"command: {' '.join(step.argv)} (cwd={cwd})", False)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(
                step.argv,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name != "nt",
                creationflags=creationflags,
            )
        except OSError as exc:
            raise ProvisionError(f"could not run {step.argv[0]}: {exc}") from exc

        # Preserve enough context for CMake compiler-identification and nested
        # call stacks: a short tail routinely discards the actual CUDA/ptxas
        # error before anyone can read it.
        tail: list[str] = []
        output: Queue[str | None] = Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for raw in process.stdout:
                    output.put(raw)
            finally:
                process.stdout.close()
                output.put(None)

        reader = Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while True:
                self._check_cancel()
                try:
                    raw = output.get(timeout=0.1)
                except Empty:
                    continue
                if raw is None:
                    break
                line = raw.rstrip("\n")
                self._on_line(progress, line, False)
                if line.strip():
                    tail.append(line.strip())
                    tail = tail[-80:]
            while process.poll() is None:
                self._check_cancel()
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    continue
            code = process.returncode
        finally:
            if process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                if process.poll() is None:
                    process.kill()
                process.wait()
            reader.join(timeout=5)
        if code != 0:
            detail = "\n".join(tail)
            raise ProvisionError(f"exit {code}\n{detail}" if detail else f"exit {code}")

    def _on_line(self, progress: StepProgress, line: str, is_stderr: bool) -> None:
        fraction = parse_progress_fraction(line)
        if fraction >= 0.0:
            progress.fraction = fraction
        progress.last_line = line
        self._log(line, is_stderr)
        self.progress_sink(progress)

    def _log(self, line: str, is_stderr: bool) -> None:
        if not line:
            return
        self.log_sink(line, is_stderr)
        if self.build_log_path is not None:
            try:
                with self.build_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{'[stderr] ' if is_stderr else ''}{line}\n")
                if self.build_log_path.stat().st_size > BUILD_LOG_RETENTION.max_bytes:
                    truncate_log(self.build_log_path, BUILD_LOG_RETENTION.tail_bytes)
            except OSError:
                pass

    def _open_build_log(self) -> None:
        if self.build_log_path is not None:
            return
        log_dir = Path(self.config.provision_dir) / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            prune_logs(log_dir, f"{self.log_prefix}-provision-*.log", BUILD_LOG_RETENTION)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = log_dir / f"{self.log_prefix}-provision-{stamp}.log"
            cfg = self.config
            header = [
                f"# Stable Imaginer {self.engine_name} provisioning attempt",
                f"platform: {cfg.platform}/{cfg.arch}",
                f"backend: {cfg.accelerator}",
                f"method: {cfg.install_method}",
                f"version: {cfg.version}",
            ]
            if cfg.accelerator == "cuda" and cfg.cuda_arch:
                header.append(f"cuda_architectures: {cuda_cmake_architectures(cfg.cuda_arch)}")
            path.write_text("\n".join(header) + "\n\n", encoding="utf-8")
            self.build_log_path = path
        except OSError:
            self.build_log_path = None

    def _check_cancel(self) -> None:
        if self.cancel_check():
            raise ProvisionCancelled(f"{self.engine_name} provisioning cancelled")
