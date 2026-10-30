"""Can a shared engine run its whole model here? (What a linked instance is told.)

``ready`` when it is loaded. ``cold`` when it is stopped but may be started for a linked
instance and the model fits in free VRAM. ``unfit`` when it would not fit. Otherwise
``unavailable``. The VRAM need is an estimate from the model files' sizes: weights, plus
room for the KV cache (llama) or activations (sd.cpp). It is advisory, never a promise.
"""

from typing import Literal

from imaginer.core.config import BackendConfig, LlamaCppBackendConfig, SdCppBackendConfig
from imaginer.domain.enums import BackendStatus
from imaginer.domain.models import EngineAvailability
from imaginer.engines import launch
from imaginer.engines.gpu import GpuMemory

MIB = 1024 * 1024
HEADROOM_MB = 1024
LLAMA_OVERHEAD = 1.2  # KV cache and compute buffers, as a share of the weights
SDCPP_ACTIVATIONS_MB = 1536

AvailabilityState = Literal["ready", "cold", "unfit", "unavailable"]


def need_mb(config: BackendConfig) -> int | None:
    """Estimated VRAM to load the model, or None when it can't be told."""
    sizes = [c.size_bytes for c in launch.check_files(config) if c.size_bytes]
    if not sizes:
        return None
    weights_mb = sum(sizes) / MIB
    if isinstance(config, LlamaCppBackendConfig):
        return round(weights_mb * LLAMA_OVERHEAD)
    if isinstance(config, SdCppBackendConfig):
        return round(weights_mb + SDCPP_ACTIVATIONS_MB)
    return None


def assess(
    *,
    status: BackendStatus,
    remote_start: bool,
    not_startable: str | None,
    need: int | None,
    gpu: GpuMemory | None,
) -> EngineAvailability:
    """``not_startable`` says why the engine can't be started here (None if it can)."""

    def verdict(state: AvailabilityState, detail: str) -> EngineAvailability:
        return EngineAvailability(
            state=state,
            detail=detail,
            remote_start=remote_start,
            vram_total_mb=gpu.total_mb if gpu else None,
            vram_free_mb=gpu.free_mb if gpu else None,
            need_mb=need,
        )

    if status is BackendStatus.OK:
        return verdict("ready", "loaded")
    if status is BackendStatus.DEGRADED:
        return verdict("cold", "loading")
    if not remote_start:
        return verdict("unavailable", "stopped, and only its owner may start it")
    if not_startable is not None:
        return verdict("unavailable", not_startable)
    if gpu is None or need is None:
        return verdict("cold", "stopped; loads on first use (fit unknown)")
    if need > gpu.free_mb - HEADROOM_MB:
        return verdict("unfit", f"needs ~{need / 1024:.1f} GB, {gpu.free_mb / 1024:.1f} GB free")
    return verdict("cold", "stopped; loads on first use")
