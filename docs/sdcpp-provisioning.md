# stable-diffusion.cpp provisioning

The `imaginer.provisioning.sdcpp` service installs `sd-server` and its companion libraries.
It shares the llama.cpp provisioner's execution, locking, progress, cancellation, staging,
and log retention. It has its own release matching, source configuration, metadata, and
update checks. Model downloads are not included. The Engines screen in the web UI runs the
same installs (`POST /v1/engines/{name}/install`) and can start an installed engine.

## Commands

```bash
uv run imaginer sdcpp plan --method source --accelerator cuda --cuda-arch 120
uv run imaginer sdcpp provision --profile windows-5090
uv run imaginer sdcpp provision --profile nas --method source --accelerator cuda --jobs 2
uv run imaginer sdcpp provision --method release --accelerator cpu --tag master-1024-77b72a2
uv run imaginer sdcpp status --profile nas
uv run imaginer sdcpp check-update --profile nas
uv run imaginer sdcpp provision --profile nas --force
```

Use the same `--profile` or `--cache-dir` for subsequent commands. The default directory
is `<data_root>/engines/stable_diffusion_cpp`. Without an explicit profile, settings come
from environment variables and built-in defaults. `--cache-dir` overrides the directory.
Do not share this directory with a llama.cpp installation.

Results are JSON on stdout. Diagnostic lines go to stderr. Errors return 1 and Ctrl+C
returns 130. A plan may probe hardware, but does not download, compile, or create the
installation directory. Status reads disk without hardware or network probes.

`auto` attempts a matching release, a source build, then a CPU release as a last resort
where one exists. Fallback is recorded in the result's `warning`, stderr, and
`active-runtime.json`'s attempt history. `--method release` and `--method source` require
the requested accelerator and do not fall back to CPU.

| Option | Meaning |
|---|---|
| `--tag` | Release tag or source git ref; `latest` is the default |
| `--accelerator` | `cpu`, `cuda`, `rocm`, `vulkan`, `metal`, `sycl-fp32`, `sycl-fp16`; detected when omitted |
| `--cuda-arch` | Compute capability such as `86`, `120`, `121`; detected when omitted |
| `--jobs` | Compile parallelism; default 2 for CUDA/ROCm and 4 for other backends |
| `--cmake-arg=-DNAME=VALUE` | Additional CMake argument; repeatable |
| `--force` | Reinstall instead of reusing a validated matching runtime |

`latest` reuses a matching local installation until `--force` is given. `check-update`
uses sd.cpp's `master-<count>-<hash>` tags, and reports any newer release without llama's
100-build notification threshold. It reports whether the update needs a download,
compilation, or is unavailable under a release-only policy. An update check never installs.

## Releases and source builds

The implementation follows upstream's
[release assets](https://github.com/leejet/stable-diffusion.cpp/releases/tag/master-1024-77b72a2),
[build instructions](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md),
and [build workflow](https://github.com/leejet/stable-diffusion.cpp/blob/master/.github/workflows/build.yml).

- Windows releases cover CPU, CUDA, Vulkan, and ROCm. CUDA needs the matching
  `cudart-sd-bin-win-cu12-x64.zip` as well as the engine ZIP.
- The verified `cuda12` family uses CUDA 12.8.1 and supports the RTX 5090's `sm_120`.
  Targets absent from that build, including `sm_121`, use the source-build path. A detected
  driver compatibility version below 12.8 also rejects that archive.
- Linux releases cover x64 CPU, Vulkan, and ROCm. Linux CUDA and ARM64 configurations
  currently need source builds. Asset availability is checked at installation time.
- macOS Metal uses the macOS archive. CPU-only macOS builds use source, since the published
  Metal archive cannot be assumed to be CPU-only.
- Unix archives' internal library symlinks are copied as regular library files. External
  links and unsafe paths are rejected. Validation runs before replacing the active binary.

Source builds need Git, CMake, and a C++ compiler. On Windows, use a Visual Studio developer
terminal with the C++ workload installed. CUDA requires a full CUDA Toolkit (`nvcc`),
ROCm requires the HIP SDK/compiler, Vulkan requires its SDK (`glslc`), and SYCL requires
the oneAPI compiler (`icpx`). Use the SDK's environment setup before running the command.
`CUDACXX` can select a particular CUDA compiler. AMD target overrides can be supplied with
`--cmake-arg=-DGPU_TARGETS=<gfx-target>` and `--cmake-arg=-DAMDGPU_TARGETS=<gfx-target>`.

The builder clones recursively and updates submodules after checking out the requested
ref. This includes the patched GGML, CUTLASS, and media dependencies. It sets
`SD_BUILD_EXAMPLES=ON`, selects the requested `SD_*` backend, and builds only the `sd-server`
target. `SD_SERVER_BUILD_FRONTEND=OFF` avoids Node/pnpm and the bundled UI; the HTTP API
is still available. CMake caches are separated by platform, architecture, and accelerator.

Installed files live under `bin/`. A replacement is validated in `bin.next/` before the
swap; the previous installation is retained in `bin.previous/`. Build logs live in
`logs/sdcpp-provision-*.log`. Source and downloads remain cached for subsequent builds.
The runtime lock is released by the OS if an installer exits or crashes.

## Backend integration

```python
from pathlib import Path
from imaginer.provisioning.sdcpp import ProvisionConfig, SdCppProvisioner

installer = SdCppProvisioner(
    ProvisionConfig(
        provision_dir=Path("/local/imaginer/engines/stable_diffusion_cpp"),
        install_method="auto",
        accelerator="cuda",
    ),
    log_sink=lambda text, is_stderr: ...,
    progress_sink=lambda progress: ...,
    cancel_check=lambda: False,
)
binary = installer.ensure_runtime()
```

Run this synchronous service in a worker thread when adding an async API or UI.
Cancellation is polled even while compiler output is silent; blocked network reads can
take up to their timeout. `install_attempts()` and `build_install_plan()` expose plans,
and `read_active_runtime()` reports the installed version, accelerator, and attempt history.

Start the returned executable with your model paths, then configure the existing `sdcpp`
HTTP adapter's `base_url`. `imaginer serve` never installs a runtime. It starts one only for a
backend with a `launch` block, when asked from the Engines screen or with `autostart: true`
(see the README's "Engines screen").
