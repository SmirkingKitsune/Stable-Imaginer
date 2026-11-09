"""Launch commands for local engines, and checks on the model files they load.

Only engines with a ``launch`` block in their backend config are started here; anything
else is remote and only its endpoint is managed. The listen address comes from the
backend's ``base_url``, so the adapter and the process always agree on it.
"""

import json
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from imaginer.backends.common import LORA_SCAN_LIMIT, expand_path
from imaginer.core.config import (
    BackendConfig,
    DiffusersBackendConfig,
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
# Where Hugging Face installs an engine's LoRAs while it has no LoRA folder, in the models
# folder. No Hub namespace starts with "_", so no repo downloads into it.
DEFAULT_LORA_FOLDER = "_loras"
# What sd-server lists as a LoRA: every file with one of these extensions, in subfolders too.
LORA_EXTENSIONS = frozenset({".safetensors", ".gguf", ".pt", ".pth", ".ckpt", ".sft"})
TOO_BIG_FOR_A_LORA = 4 * 1024**3


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
        checks.append(check_file(key, flag, what, Path(raw) if raw else None, engine=config.type))
    return checks


@dataclass(frozen=True, slots=True)
class LoraFolderCheck:
    path: str | None
    exists: bool = False
    count: int | None = None
    problem: str | None = None
    warning: str | None = None


def default_lora_dir(settings: Settings) -> Path:
    return settings.models_dir / DEFAULT_LORA_FOLDER


def configured_lora_dir(config: SdCppBackendConfig | DiffusersBackendConfig) -> Path | None:
    """The engine's LoRA folder as set: an sd-server's --lora-model-dir, a Diffusers
    backend's lora_dir."""
    if isinstance(config, DiffusersBackendConfig):
        return config.lora_dir
    return config.launch.lora_model_dir if config.launch is not None else None


def lora_dir(config: SdCppBackendConfig | DiffusersBackendConfig, settings: Settings) -> Path:
    """Where the engine's LoRAs are: its LoRA folder, else where they would go."""
    configured = configured_lora_dir(config)
    return expand(configured) if configured is not None else default_lora_dir(settings)


expand = expand_path


def check_lora_folder(
    path: Path | None, extensions: frozenset[str] = LORA_EXTENSIONS
) -> LoraFolderCheck:
    """An engine's LoRA folder: how many LoRAs it lists from it (files with one of
    ``extensions``), and anything in it that isn't one (a whole diffusion model in it
    would be listed as a LoRA too)."""
    if path is None:
        return LoraFolderCheck(None)
    folder = expand(path)
    shown = str(path)
    if not folder.exists():
        return LoraFolderCheck(shown, problem="folder not found")
    if not folder.is_dir():
        return LoraFolderCheck(shown, exists=True, problem="a file, not a folder")
    count = seen = 0
    biggest: tuple[int, str] = (0, "")
    for root, dirs, names in os.walk(folder, onerror=lambda _: None):
        dirs.sort()
        seen += len(dirs) + len(names)
        if seen > LORA_SCAN_LIMIT:
            crowded = f"more than {LORA_SCAN_LIMIT} files: keep only LoRAs in this folder"
            return LoraFolderCheck(shown, exists=True, warning=crowded)
        for name in names:
            if Path(name).suffix.lower() not in extensions:
                continue
            count += 1
            try:
                size = (Path(root) / name).stat().st_size
            except OSError:
                continue
            biggest = max(biggest, (size, name))
    warning: str | None = None
    if biggest[0] >= TOO_BIG_FOR_A_LORA:
        warning = (
            f"{biggest[1]} ({biggest[0] / 1e9:.1f} GB) is too big for a LoRA, but every "
            "weights file here is listed as one: keep only LoRAs in this folder"
        )
    return LoraFolderCheck(shown, exists=True, count=count, warning=warning)


def check_file(
    key: str, flag: str, what: str, path: Path | None, *, engine: str | None = None
) -> FileCheck:
    """``engine`` adds what that engine can't load: llama-server reads GGUF only, from a split
    GGUF's first shard; sd-server reads sharded safetensors through their index (a
    ``*.safetensors.index.json``, or a folder holding ``model.safetensors.index.json``), so
    one shard on its own would load part of a model."""
    if path is None:
        return FileCheck(key, flag, what, None)
    expanded = Path(os.path.expandvars(path)).expanduser()
    shown = str(path)
    if expanded.is_dir():
        index = expanded / ROOT_INDEX
        if engine == "sdcpp" and index.is_file() and not (expanded / "model_index.json").exists():
            return _check_index(key, flag, what, shown, index)
        return FileCheck(key, flag, what, shown, exists=True, problem=_folder_problem(expanded))
    if not expanded.is_file():
        return FileCheck(key, flag, what, shown, problem="file not found")
    if expanded.name.endswith(".json"):
        if engine == "llamacpp":
            return FileCheck(key, flag, what, shown, exists=True, problem=GGUF_ONLY)
        return _check_index(key, flag, what, shown, expanded)
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
    elif engine == "llamacpp" and fmt == "safetensors":
        problem = GGUF_ONLY
    else:
        problem = _shard_problem(expanded, fmt, engine)
    return FileCheck(
        key, flag, what, shown, exists=True, size_bytes=size, format=fmt, problem=problem
    )


GGUF_ONLY = "llama-server loads GGUF files only"
ROOT_INDEX = "model.safetensors.index.json"
_SHARD = re.compile(r"^(?P<base>.+)-(?P<n>\d{5})-of-(?P<total>\d{5})\.(?:gguf|safetensors)$")


def _shard_problem(path: Path, fmt: FileFormat, engine: str | None) -> str | None:
    shard = _SHARD.match(path.name)
    if shard is None:
        return None
    n, total = int(shard["n"]), int(shard["total"])
    if fmt == "gguf":
        if n != 1:
            first = f"{shard['base']}-00001-of-{shard['total']}.gguf"
            return f"shard {n} of {total}: point at the first one, {first}"
        return None
    if engine != "sdcpp":
        return None
    index = path.with_name(f"{shard['base']}.safetensors.index.json")
    if index.is_file():
        return f"one of {total} shards: point at {index.name} to load them all"
    return f"one of {total} shards, with no {index.name} beside it to load them all"


def _folder_problem(folder: Path) -> str:
    if (folder / "model_index.json").is_file():
        return (
            "a Diffusers pipeline folder: point at its transformer, vae or text_encoder "
            "weights (the .safetensors.index.json where they are sharded)"
        )
    return "a folder, not a model file"


def _check_index(key: str, flag: str, what: str, shown: str, index: Path) -> FileCheck:
    """A safetensors index: its weight_map names the shards, which must all be beside it."""
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    weight_map = data.get("weight_map") if isinstance(data, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        problem = "not a safetensors index (no weight_map)"
        return FileCheck(key, flag, what, shown, exists=True, problem=problem)
    shards = sorted({str(name) for name in weight_map.values()})
    size = 0
    missing = []
    for name in shards:
        try:
            size += (index.parent / name).stat().st_size
        except OSError:
            missing.append(name)
    if missing:
        problem = f"{len(missing)} of its {len(shards)} shards are missing, e.g. {missing[0]}"
        return FileCheck(key, flag, what, shown, exists=True, problem=problem)
    return FileCheck(key, flag, what, shown, exists=True, size_bytes=size, format="safetensors")


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
    if isinstance(launch, SdCppLaunch):
        argv += _lora_args(launch)
    if isinstance(config, SdCppBackendConfig):
        argv += ["--listen-ip", host, "--listen-port", str(port)]
    else:
        argv += ["--host", host, "--port", str(port)]
    return argv + list(launch.extra_args)


def _lora_args(launch: SdCppLaunch) -> list[str]:
    args = []
    if launch.lora_model_dir is not None:
        folder = expand(launch.lora_model_dir)
        if not folder.is_dir():
            raise LaunchError(f"--lora-model-dir {launch.lora_model_dir}: folder not found")
        args += ["--lora-model-dir", str(folder)]
    if launch.lora_apply_mode is not None:
        args += ["--lora-apply-mode", launch.lora_apply_mode]
    return args
