# Writing a backend adapter

The orchestrator only knows two protocols, both in `src/imaginer/backends/base.py`:

- `LLMBackend` for chat models.
- `ImageBackend` for image models.

An adapter turns one engine into one of these. Routing, jobs, tools, storage and the HTTP API
work unchanged on top of it.

Shipped adapters:

| Type | Module | Engine | Transport |
|---|---|---|---|
| `llamacpp` | `backends/llamacpp.py` | llama.cpp `llama-server` | OpenAI chat completions over HTTP (SSE) |
| `sdcpp` | `backends/sdcpp.py` | stable-diffusion.cpp `sd-server` | native `/sdcpp/v1` async jobs over HTTP |
| `diffusers` | `backends/diffusers_inproc.py` | Hugging Face Diffusers | in-process, worker thread |

## The contract

### Identity

| Member | Meaning |
|---|---|
| `name` | The config key, e.g. `sdcpp`. |
| `model` | The engine's model id. It is recorded on every message and job, so name it precisely. |
| `max_ref_images` | Image backends only: how many input images `edit` accepts, a mask included. |

### Capabilities

`capabilities() -> set[Capability]` must be **synchronous and cheap**: return a cached set.

- The set is the **declared** capabilities (`declared_capabilities` in config, defaulting to
  everything the adapter supports) **narrowed by the last probe**.
- A probe may only remove capabilities, never add them.
- An unprobed or unreachable backend advertises nothing.
- Never advertise something the engine can't do. If the engine can't report a capability
  (sd-server's command-line mode reports every feature as available), keep it declared-only
  and document that the profile must restrict it.

The registry routes on capabilities:

- A missing capability is a 422 `capability_unavailable`, never a fallback to another backend.
- `T2I` and `EDIT` select the task.
- `MULTI_REF` is required for more than one input image. A mask counts as an input.
- `RGBA` is required for `alpha=True`.
- `VISION_IN` is required for image attachments in chat.
- `TOOLS` decides whether tools are offered.

### Health

`async health() -> BackendHealth` probes the engine and **refreshes the cached capabilities**.
Keep it under the registry's 5 s timeout.

| Status | Meaning | Effect on requests |
|---|---|---|
| `ok` | Ready, or able to queue work (e.g. a lazy model load) | Routed |
| `degraded` | Reachable but not ready (model loading on the engine) | 503 `engine_unavailable` |
| `down` | Unreachable or unusable (no CUDA, load failed) | 503 `engine_unavailable` |

`unavailable` is set by the registry, not by adapters. It means the backend is disabled, or its
factory raised (for example, an optional extra isn't installed).

Before routing, the registry re-probes a backend whose last good probe is older than
`orchestration.probe_ttl_s`. A backend that isn't `ok` is re-probed on every request, so a
recovered engine is picked up immediately.

### Image requests

`generate(GenRequest)` and `edit(EditRequest)` produce **one image per call**. The job
manager loops for `n`, so every output has its own seed.

- **Seed:** `req.seed` is always set. Pass it to the engine, and return the seed the engine
  actually used in `ImageResult.seed` (read it back if the engine reports it).
- **Sizes:** use `backends/common.resolve_params`. It snaps to the backend's `size_multiple`,
  derives sizes from `aspect_ratio`, and makes an edit's output follow the first input's shape
  (`base_size`). Record the result in `ImageResult.params`, so the job stores what really ran.
- **Transparency:** apply `styled_prompt`. Qwen-Image-2.1's transparency is prompt-driven.
  Transparent output must be PNG or WebP.
- **Edits:** use `prepare_inputs`. It validates the count, downscales inputs to
  `max_ref_pixels`, and puts the mask last. If there is a mask, append `mask_instruction` with
  `mask_prompt`.
- **Progress:** if `req.on_progress` is set, call it with `GenProgress(step, total_steps)` as
  the engine reports steps. Make sure in-flight progress calls have finished before returning.
- **Engine job id:** if the engine has job ids, call `req.on_submitted(engine_job_id)` once it's
  known. It's stored on the job for debugging.
- **Errors:** raise `EngineUnavailableError` (503), `EngineTimeoutError` (504) or `EngineError`
  (502). Never let a raw library exception escape. `backends/openai_http/transport.py` maps
  HTTP failures and applies retries: only requests that never reached the engine, or a 503, are
  retried for non-idempotent calls.

### Cancellation

When the job is cancelled, the adapter's coroutine receives `asyncio.CancelledError`. It must:

1. Stop the engine's work, not just stop waiting for it. sd-server gets
   `POST /sdcpp/v1/jobs/{id}/cancel`. Diffusers raises from `callback_on_step_end` on the next step.
2. Not return until the GPU is actually free (Diffusers awaits its worker thread). The job
   manager's per-backend semaphore is released only then.
3. Re-raise `CancelledError`.

Also handle cancellation that arrives while the job is still being submitted: the engine may
already have accepted it (see `SdCppBackend._submit`).

### Lifecycle

- Construct cheaply: no network calls, no model loading. The factory may fail fast on missing
  dependencies, and the registry shows the reason.
- `aclose()` releases clients, threads and GPU memory.
- Heavy dependencies must be imported **inside** functions, never at module level.
  `tests/backends/test_diffusers.py` checks that every backend module imports with `torch`
  blocked.

## Adding a backend type

1. **Config:** add a pydantic config class in `core/config.py` with `type: Literal["mytype"]`,
   add it to the `BackendConfig` union, and add it to `BACKEND_KINDS`.
2. **Adapter:** write it in `backends/<mytype>.py`.
3. **Registration:** register a factory in `backends/registry.py` `FACTORIES`.
4. **Fake engine:** add a fake that speaks the engine's wire format in `tests/fakes/` and serve it
   with `LiveServer`. Test against it: probing, request mapping, progress, errors, and
   cancellation, including cancellation during submission.
5. **Integration tests:** add opt-in tests in `tests/integration/`, marked `integration` and
   skipped unless an env var points at a real engine.

## Design note: an SGLang Diffusion adapter (not implemented)

This targets SGLang on the Linux hosts (the NAS now, the DGX Spark later). It would be
`type: sglang_diffusion`, an `ImageBackend` over SGLang's
[OpenAI-compatible images API](https://docs.sglang.io/diffusion/api/openai_api.html), started
with `sglang serve --model-path Qwen/Qwen-Image-2.1`.
What the docs show, as of 2026-10-28:

| Need | SGLang Diffusion | Adapter plan |
|---|---|---|
| Probe / model id | `GET /v1/models`: `id`, `supported_task_types`, `pipeline_class` | `health()`. Map `t2i` to `T2I` and `i2i` to `EDIT`; keep `MULTI_REF` and `RGBA` declared-only until verified. |
| Generate | `POST /v1/images/generations` (JSON: `prompt`, `size`, `n`, `response_format=b64_json`; model fields via `extra_body`) | `n=1`, `size=WxH` from `resolve_params`, `response_format=b64_json` |
| Edit | `POST /v1/images/edits` (multipart: `image`, `prompt`, `size`) | Send the inputs from `prepare_inputs`. Multiple `image` parts are unverified. |
| Seed, steps, guidance, negative prompt | Not documented on the API page | Try `extra_body`. If the seed can't be set, report the engine's seed, or note that outputs aren't reproducible. |
| Progress | None for images | No `on_progress` calls; `job_update` shows only the status |
| Cancel | No cancel for images. Deleting a job record doesn't stop the GPU. | On `CancelledError`, close the request and re-raise. Document that the GPU finishes the image regardless. |

**Open questions to settle against a live server before implementing:**

- Seed control.
- Multi-image edits.
- Alpha output.
- Whether an image request can be cancelled at all.

Until then the shipped adapters are the reference: `sdcpp` for remote engines and `diffusers`
for in-process ones.

## The `linked` type

`type: linked` is not an engine adapter of its own kind: it uses an engine that a linked
Imaginer instance shares, whatever its type there, over that instance's peer API
([linked-instances.md](linked-instances.md)). Its factory is registered by the link service at
startup, its health and capabilities come from the peer's status, and its config says which
`kind` (`llm` or `image`) it is, since routing must know before the peer answers. A new adapter
type becomes shareable without any change here.
