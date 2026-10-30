"""Launch commands for local engines, and checks on the model files they load.

Only engines with a ``launch`` block in their backend config are started here; anything
else is remote and only its endpoint is managed. The listen address comes from the
backend's ``base_url``, so the adapter and the process always agree on it.
"""

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from imaginer.core.config import (
    BackendConfig,
    LlamaCppBackendConfig,
    LlamaCppLaunch,
    SdCppBackendConfig,
    SdCppLaunch,
    Settings,
)

FileFormat = Literal["gguf", "safetensors", "unknown"]

# (config field, command-line flag, what it is) in launch order.
SDCPP_FILES: tuple[tuple[str, str, str], ...] = (
    ("diffusion_model", "--diffusion-model", "diffusion model"),
    ("vae", "--vae", "VAE"),
    ("llm", "--llm", "text encoder"),
    ("llm_vision", "--llm_vision", "vision projector"),
)
LLAMA_FILES: tuple[tuple[str, str, str], ...] = (
    ("model", "-m", "model"),
    ("mmproj", "--mmproj", "vision projector"),
)
# Engine type -> the folder its provisioner installs into, under <data_root>/engines.
RUNTIME_FOLDERS = {"sdcpp": "stable_diffusion_cpp", "llamacpp": "llama_cpp"}


class LaunchError(ValueError):
    """The engine can't be started as configured."""


@dataclass(frozen=True, slots=True)
class FileCheck:
    key: str
    flag: str
    what: str
    path: str | None
    exists: bool = False
    size_bytes: int | None = None
    format: FileFormat | None = None
    problem: str | None = None

    @property
    def ok(self) -> bool:
        return self.path is not None and self.problem is None


def endpoint(config: BackendConfig) -> tuple[str, int]:
    """Host and port from the backend's base_url."""
    url = urlsplit(getattr(config, "base_url", "") or "")
    port = url.port or (443 if url.scheme == "https" else 80)
    return url.hostname or "127.0.0.1", port


def is_loopback(host: str) -> bool:
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def file_specs(config: BackendConfig) -> tuple[tuple[str, str, str], ...]:
    if isinstance(config, SdCppBackendConfig):
        return SDCPP_FILES
    if isinstance(config, LlamaCppBackendConfig):
        return LLAMA_FILES
    return ()


def check_files(config: BackendConfig) -> list[FileCheck]:
    """Every model file the launch block names (or leaves empty), checked on disk."""
    launch = getattr(config, "launch", None)
    checks = []
    for key, flag, what in file_specs(config):
        raw = getattr(launch, key, None) if launch is not None else None
        checks.append(check_file(key, flag, what, Path(raw) if raw else None))
    return checks


def check_file(key: str, flag: str, what: str, path: Path | None) -> FileCheck:
    if path is None:
        return FileCheck(key, flag, what, None)
    expanded = Path(os.path.expandvars(path)).expanduser()
    shown = str(path)
    if not expanded.is_file():
        return FileCheck(key, flag, what, shown, problem="file not found")
    try:
        size = expanded.stat().st_size
        with expanded.open("rb") as fh:
            head = fh.read(16)
    except OSError as exc:
        return FileCheck(key, flag, what, shown, exists=True, problem=f"unreadable: {exc}")
    fmt = sniff_format(head, size)
    problem = None
    if fmt == "unknown":
        problem = "not a GGUF or safetensors file"
    return FileCheck(
        key, flag, what, shown, exists=True, size_bytes=size, format=fmt, problem=problem
    )


def sniff_format(head: bytes, size: int) -> FileFormat:
    """GGUF starts with b"GGUF"; safetensors with a little-endian u64 header length
    followed by a JSON object."""
    if head[:4] == b"GGUF":
        return "gguf"
    if len(head) >= 9:
        (length,) = struct.unpack("<Q", head[:8])
        if 2 <= length < min(size, 100 * 1024 * 1024) and head[8:9] == b"{":
            return "safetensors"
    return "unknown"


def runtime_dir(settings: Settings, engine_type: str) -> Path | None:
    folder = RUNTIME_FOLDERS.get(engine_type)
    return settings.data_root / "engines" / folder if folder else None


def executable(
    settings: Settings, config: BackendConfig, active_runtime: Callable[[Path], dict[str, object]]
) -> Path | None:
    """The configured executable, else the managed runtime's, else None."""
    launch = getattr(config, "launch", None)
    if launch is not None and launch.executable is not None:
        return Path(os.path.expandvars(launch.executable)).expanduser()
    root = runtime_dir(settings, config.type)
    if root is None:
        return None
    found = active_runtime(root).get("executable")
    return Path(str(found)) if found else None


def command(config: BackendConfig, exe: Path) -> list[str]:
    """The full argv. Raises LaunchError when a required file is missing."""
    launch: SdCppLaunch | LlamaCppLaunch | None = getattr(config, "launch", None)
    if launch is None:
        raise LaunchError("this engine has no launch settings; it is managed elsewhere")
    host, port = endpoint(config)
    checks = check_files(config)
    required = {"diffusion_model"} if isinstance(config, SdCppBackendConfig) else {"model"}
    for check in checks:
        if check.key in required and check.path is None:
            raise LaunchError(f"{check.flag} ({check.what}) is not set")
        if check.path is not None and check.problem is not None:
            raise LaunchError(f"{check.flag} {check.path}: {check.problem}")
    argv = [str(exe)]
    for check in checks:
        if check.path is not None:
            argv += [check.flag, str(Path(os.path.expandvars(check.path)).expanduser())]
    if isinstance(config, SdCppBackendConfig):
        argv += ["--listen-ip", host, "--listen-port", str(port)]
    else:
        argv += ["--host", host, "--port", str(port)]
    return argv + list(launch.extra_args)
