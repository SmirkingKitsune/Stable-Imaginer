# llama.cpp provisioning

The UI-independent service in `imaginer.provisioning.llamacpp` is adapted from
[Batch-Image-Interrogator's provisioner](https://github.com/SmirkingKitsune/Batch-Image-Interrogator/blob/main/core/llama_provisioner.py).
Its MIT notice ships alongside the module. No Qt, Electron, or additional Python
dependencies are required. Engine process management and model downloads are separate
from provisioning; the current llama.cpp adapter continues to talk to `base_url`.

## CLI

```bash
# Inspect a source build without downloading or compiling anything.
uv run imaginer llama plan --method source --accelerator cuda --cuda-arch 120

# Matched release, source fallback, then CPU release fallback.
uv run imaginer llama provision --profile nas

# Require a CUDA source build; no CPU fallback.
uv run imaginer llama provision --method source --accelerator cuda --cuda-arch 121 --jobs 2

# Require a specific release.
uv run imaginer llama provision --method release --tag b11970 --accelerator cpu

# Inspect and update the same installation directory.
uv run imaginer llama status --cache-dir /path/to/runtime
uv run imaginer llama check-update --cache-dir /path/to/runtime --accelerator cuda
uv run imaginer llama provision --cache-dir /path/to/runtime --accelerator cuda --force
```

All commands print JSON to stdout. Provisioning logs stream to stderr, with failure
details and the build log path. Failures return exit code 1; Ctrl+C returns 130.
`status` inspects disk without hardware probes or network requests. `plan` may probe
hardware but does not create the installation directory, download, or compile.
`check-update` queries GitHub and inherits the original 100-build notification threshold;
its JSON includes the build distance even below that threshold.

Options shared by these commands:

| Option | Meaning |
|---|---|
| `--profile`, `--config-dir` | Resolve the data root using an existing Imaginer profile |
| `--cache-dir` | Override `<data_root>/engines/llama_cpp` |
| `--method auto/release/source` | Acquisition strategy |
| `--tag` | Release tag or source ref; default `latest` |
| `--accelerator` | `cuda`, `rocm`, `vulkan`, `metal`, `sycl-fp32`, `sycl-fp16`, `openvino`, `cpu`; detected when omitted |
| `--cuda-arch` | Compute capability, e.g. `86`, `120`, `121`, or a semicolon-separated list |
| `--release-variant` | Restrict CUDA releases to `cuda-12` or `cuda-13` |
| `--jobs` | Compiler parallelism; 0 chooses 2 for CUDA/ROCm and 4 otherwise |
| `--cmake-arg=-DNAME=VALUE` | Extra CMake argument; repeatable |
| `--force` | Replace a cached installation |

Without an explicit profile, commands use settings from the environment and built-in
defaults, so they also work outside a source checkout. Supplying a profile uses the usual
YAML/environment precedence. Reuse in `auto` requires matching platform, architecture,
accelerator, version pin, CUDA targets, and extra build arguments, plus executable validation.
`latest` reuses a matching installation until `--force` is given.

## Source builds and releases

Release selection matches OS, architecture, accelerator, and CUDA driver compatibility.
Windows CUDA installs include the base archive and matching CUDA runtime DLL archive.
Linux and macOS use their matching tar archives. Upstream release availability is discovered
from GitHub rather than assuming every accelerator is published for every architecture.

Source builds need Git, CMake, and a C++ toolchain. On Windows, use a Visual Studio developer
terminal with the C++ workload installed. CUDA additionally requires the CUDA Toolkit;
having only a working NVIDIA driver is insufficient. The provisioner checks `nvcc` targets
and performs a small compile probe before starting the build. `CUDACXX` selects a particular
CUDA compiler. ROCm needs `hipcc`, Vulkan needs `glslc`, SYCL needs `icpx`, and OpenVINO
needs its toolkit environment. The source checkout is managed by the provisioner; do not
put independent source edits there.

Each platform/architecture/accelerator has its own build tree. CUDA reconfiguration clears
cached compiler selection. Installation stages the executable with its shared libraries,
validates it, and swaps it into `bin/`, retaining `bin.previous/`. An OS lock prevents
concurrent installations into the same root. Archive paths are checked before extraction;
tar extraction uses Python's data filter.

The directory contains `bin/`, `active-runtime.json`, release downloads, `llama.cpp-src/`,
and `logs/llama-provision-*.log`. Runtime metadata records the actual accelerator and
the attempts that led to it. Build logs are capped and older logs are pruned separately.

## Python service

```python
from pathlib import Path
from imaginer.provisioning.llamacpp import LlamaProvisioner, ProvisionConfig

provisioner = LlamaProvisioner(
    ProvisionConfig(provision_dir=Path("/local/imaginer/engines/llama_cpp")),
    log_sink=lambda text, is_stderr: ...,  # streamed diagnostic lines
    progress_sink=lambda progress: ...,  # step, total, label, fraction, last_line
    cancel_check=lambda: False,  # read a threading.Event in a UI worker
)
binary = provisioner.ensure_runtime()
```

The service is synchronous; run it in a worker thread when integrating with an async API
or UI. `install_attempts()` and `build_install_plan()` expose the planned work without
executing it. `read_active_runtime()` exposes installed metadata and `check_for_update()`
describes whether an update needs a download or compilation.

Cancellation is polled while commands are silent as well as during output. Downloads and
in-process actions check cancellation at progress boundaries; a blocked network read can
take up to its timeout. Compilation subprocess trees are terminated on cancellation.
Progress fractions refer to the current CMake step, not a predicted overall duration.
