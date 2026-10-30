"""GPU memory, as nvidia-smi reports it. Other accelerators report nothing (None)."""

import shutil
import subprocess
import time
from dataclasses import dataclass

CACHE_S = 5.0
_cache: tuple[float, "GpuMemory | None"] | None = None


@dataclass(frozen=True, slots=True)
class GpuMemory:
    """The first GPU's memory in MiB: the one engines load onto unless told otherwise."""

    name: str | None
    total_mb: int
    used_mb: int

    @property
    def free_mb(self) -> int:
        return max(self.total_mb - self.used_mb, 0)


def gpu_memory() -> GpuMemory | None:
    """Blocking (it runs a process): call it in a thread. Cached for a few seconds."""
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < CACHE_S:
        return _cache[1]
    result = _query()
    _cache = (now, result)
    return result


def _query() -> GpuMemory | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse(done.stdout)


def parse(output: str) -> GpuMemory | None:
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            return GpuMemory(name=parts[0] or None, total_mb=int(parts[1]), used_mb=int(parts[2]))
        except ValueError:
            continue
    return None
