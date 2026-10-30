# Stable-Imaginer

Backend for a local, single-user multimodal chat app. A FastAPI orchestrator owns
conversations and their artifacts, and talks to separate inference engines over HTTP:

- **Chat LLM:** `llama.cpp`'s `llama-server`. The reference model is Qwen3-VL-8B-Instruct, with vision and tool calling.
- **Images:** `stable-diffusion.cpp`'s `sd-server` running Qwen-Image-2.1, reached through its native `/sdcpp/v1` API.
- **Optional:** an in-process Hugging Face Diffusers backend, used as the reference implementation.

Every artifact (upload, generated image, edit) belongs to exactly one conversation. It is
stored under that conversation's folder, and deleting the conversation removes the artifact.
Engines sit behind adapters, so the orchestrator doesn't care which engine or OS serves a request.

Runs natively on Windows 11 and Linux (x64 and aarch64). No WSL or Docker needed.

> **Status: all 5 milestones done, plus a browser UI.** Conversations, uploads, artifact
> storage, streaming chat with a llama.cpp model (including image input), and tool-driven image
> generation and editing all work, on sd-server or in-process with Diffusers. The UI at `/` uses
> only this HTTP API.

## Setup

Requires Python 3.12. [uv](https://docs.astral.sh/uv/) is recommended. Plain `pip` works too.

**Windows (PowerShell)**

```powershell
winget install --id=astral-sh.uv -e
git clone https://github.com/SmirkingKitsune/Stable-Imaginer.git
cd Stable-Imaginer
uv sync
uv run imaginer serve --profile dev
```

**Linux (bash)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/SmirkingKitsune/Stable-Imaginer.git
cd Stable-Imaginer
uv sync
uv run imaginer serve --profile dev
```

Then open <http://127.0.0.1:8700/> for the UI, or <http://127.0.0.1:8700/docs> for the
interactive OpenAPI docs.

The `dev` profile expects the fake engines, which need no GPU. Start them in a second terminal:

```bash
uv run python -m tests.fakes
```

The fake chat model echoes your message back, with two exceptions:

- `draw ...` calls `generate_image`. The fake sd-server paints a solid colour derived from the seed.
- `edit img-1: ...` calls `edit_image` on that image.

Without uv: `python -m venv .venv`, activate it, then `pip install -e . --group dev`
(needs pip 25.1 or newer).

## Configuration

Each host gets one YAML profile in `config/`. The usual setup is **standalone**: one computer
runs the server and both engines, and keeps every conversation. Start from `standalone.yaml`.

| Profile | Host | Role |
|---|---|---|
| `standalone` | any | Both engines on this computer, launched from the Engines screen. Start here |
| `dev` | any | Points at fake engines, so no GPU is needed |
| `dev-peer` | any | A second dev instance (port 8701) for trying [linked instances](#linked-instances) |
| `windows-5090` | Windows 11, RTX 5090 | Linked with `nas`: images here, chat on the NAS's engine |
| `nas` | Linux, RTX 2000e Ada | Linked with `windows-5090`: chat here, images on the desktop's engine |

Pick a profile with `--profile <name>` (or a path to a YAML file) or `IMAGINER_PROFILE`.
Any setting can be overridden from the environment with the `IMAGINER_` prefix and `__`
for nesting, for example `IMAGINER_SERVER__PORT=9000` or `IMAGINER_DATA_ROOT=D:/Imaginer`.

**Secrets never go in YAML.** Set them in the environment, or in a gitignored `.env`
(see `.env.example`):

- `IMAGINER_SERVER__AUTH_TOKEN`: if set, every `/v1` request needs `Authorization: Bearer <token>`.
  The server binds to `127.0.0.1` by default and warns if you bind it more widely without a token.
  Linking with other instances over the network requires it.
- Engine API keys: each backend's `api_key_env` names the env var that holds its key.

Sampler defaults (steps, CFG, sampler, sigmas, size) are **per backend**, because engines
disagree. For Qwen-Image-2.1, the sd.cpp docs use CFG 6.0 with euler, while Diffusers samples
without guidance (CFG 1.0) at 40 steps.

More image settings:

- **Sizes** are rounded to `size_multiple` (default 32, which Qwen-Image-2.1 requires).
  An `aspect_ratio` keeps the default image's pixel area.
- **Sigmas:** `defaults.sigmas` accepts a list of numbers or the preset
  `qwen-image-2.1-turbo`, for the Turbo checkpoint: 8 steps, CFG 1.0.
- **Progress:** an sdcpp backend's `preview` (default `proj`) is how sd-server reports step
  progress. With `none` there is no step progress, only the job status.

Useful commands:

```bash
uv run imaginer config show --profile windows-5090
```

```bash
uv run imaginer db upgrade --profile nas
```

```bash
uv run imaginer backends check --profile windows-5090
```

`backends check` probes every engine and prints its status and capabilities. It exits with
code 1 if an enabled backend isn't healthy. The server also migrates the database on startup
unless `db.auto_migrate: false`.

## Linked instances

Two (or more) Imaginer instances on one network can be **linked**. That is optional: a
standalone instance needs nothing from anyone. Linking helps when another computer has the
GPU for a model this one can't hold:

- **Offloading:** this computer keeps its conversations, and one or more of its tasks (chat,
  t2i, edit) use an engine that a linked instance **shares**. The whole model runs over there;
  requests and results travel through the two servers, and images are stored here.
- **One gallery:** each instance's gallery also shows the conversations of the instances it is
  linked with, marked with their name. They stay where they are. Opening one, and continuing
  it, goes through this server to its home, where the turn runs with that instance's engines.

To link two instances:

1. On both, set `links.enabled: true`, make the server reachable (`server.host: 0.0.0.0`) and
   set `IMAGINER_SERVER__AUTH_TOKEN`. The server refuses to start otherwise.
2. On one, open **Linked instances** (in the engine status menu) and press **Allow a link**, or
   run `uv run imaginer link code --profile <profile>` on a headless host. It shows a code,
   valid for 10 minutes and once.
3. On the other, enter the first one's address and the code. Instances on the same network
   are listed there too (mDNS).
4. Share engines (`links.share`), and on the other side press **Use for chat** or **Use for
   images**: that adds a `type: linked` backend and routes the task to it. Both are saved to
   `config/<profile>.local.yaml`.

```yaml
backends:
  llama_nas:
    type: linked
    link: nas        # this instance's name for the link
    engine: llama    # the backend's name on the linked instance
    kind: llm
routing:
  chat: llama_nas
links:
  enabled: true
  share:
    sdcpp: {remote_start: true}  # may be started for a linked instance, if the model fits
```

For each engine it shares, an instance reports whether it can hold the whole model: **ready**
(loaded), **loads on use** (stopped, startable, and it fits in free VRAM), **won't fit**, or
unavailable. Linked engines never fall back silently: if the other instance is off, the request
fails with `engine_unavailable`, as for any engine that is down. Engines stay on `127.0.0.1`:
linked instances reach them through the server, with the token they were issued when they paired.
See [docs/linked-instances.md](docs/linked-instances.md) for the protocol and security model.

## Data layout

```
<data_root>/
  imaginer.sqlite3                       # SQLite in WAL mode
  conversations/<conversation_id>/
    artifacts/<artifact_id>.<ext>
```

`data_root` defaults to `%LOCALAPPDATA%\imaginer` on Windows and `~/.local/share/imaginer`
on Linux. Keep it on a local disk: SQLite WAL mode does not work on network shares.

Deleting a conversation marks it deleted first, then removes its folder, then its rows. If
removing the files fails (on Windows, a file that is open elsewhere can't be deleted), the
conversation stays hidden and a background sweep finishes the delete later.

## API overview (v1)

| Method | Path | Notes |
|---|---|---|
| `POST`, `GET` | `/v1/conversations` | list is newest-activity first; cursor pagination; each item adds `image_count`, `cover` (latest image) and `turn_running` |
| `GET` | `/v1/conversations/search?q=&fields=&include_archived=` | searches titles, your prompts, image prompts, labels and replies, one page of conversations at a time; each hit has a `match` (field, snippet), each page `scanned` and `total` |
| `GET`, `PATCH`, `DELETE` | `/v1/conversations/{id}` | `PATCH {"archived": true}` archives |
| `GET` | `/v1/conversations/{id}/messages?after_seq=&leaf_id=&all_branches=` | one branch in `seq` order: the latest, or the one ending at `leaf_id`; `all_branches=true` returns every message |
| `POST` | `/v1/conversations/{id}/messages` | send a message; the reply streams back as SSE (below). `parent_id` continues a branch, `edit_of` forks one |
| `GET` | `/v1/conversations/{id}/branches` | every branch: name, kind, fork point, message count, latest image |
| `POST` | `/v1/conversations/{id}/messages/{tool_message_id}/retry` | run a tool call again (new seed) as a new attempt on a new branch; streams as SSE |
| `GET` | `/v1/conversations/{id}/turn` | the running turn, or 404 |
| `GET` | `/v1/conversations/{id}/turn/events` | re-attach to the running turn's SSE; honours `Last-Event-ID` |
| `POST` | `/v1/conversations/{id}/turn/cancel` | stop the running turn; text so far is kept, its jobs are cancelled |
| `GET` | `/v1/conversations/{id}/jobs?status=` | generation jobs, oldest first |
| `GET` | `/v1/jobs/{id}` | one job: requested and resolved parameters, per-output seeds, progress |
| `POST` | `/v1/jobs/{id}/cancel` | cancel a queued or running job and its engine job; 409 if it already finished |
| `GET` | `/v1/conversations/{id}/artifacts?kind=&source=` | |
| `POST` | `/v1/conversations/{id}/artifacts?kind=&mask_for=` | multipart `file`; type is detected from the bytes. `kind=mask` stores a painted mask (white = change) labelled `mask-N` |
| `GET` | `/v1/artifacts/{id}` | metadata, including its per-conversation `label` (`img-3`) |
| `GET` | `/v1/artifacts/{id}/content` | bytes; `ETag` = sha256; supports `If-None-Match` |
| `GET` | `/v1/artifacts/{id}/lineage` | parents (`edit_source`, `reference`, `mask`) and children |
| `GET` | `/v1/backends` | health, capabilities, `model`, `default_for` (routed tasks) and, for image backends, `image_defaults` |
| `GET`, `PATCH` | `/v1/engines`, `/v1/engines/{name}` | the Engines screen: endpoint, capabilities, launch files (checked on disk), process and runtime. `PATCH` saves to `config/<profile>.local.yaml` |
| `POST` | `/v1/engines/{name}/start`, `/stop` | start or stop a local engine that has `launch` settings |
| `GET` | `/v1/engines/{name}/log`, `/log/file`, `/events` | the last run's output; `/events` streams `line`, `state` and `install` as SSE |
| `POST` | `/v1/engines/{name}/install/plan`, `/install`, `/install/cancel` | install the engine's runtime in the background, as `imaginer sdcpp provision` would |
| `GET` | `/v1/gallery?limit=&cursor=&include_archived=` | this instance's conversations and every linked instance's, newest activity first; each item has an `origin` (`base_path` reaches a remote one), each page `sources` with their status |
| `GET`, `POST` | `/v1/links`, `/v1/links/discovered` | the Links screen: this instance, its links and what each shares (with VRAM fit); `POST {url, code, name?}` pairs |
| `POST` | `/v1/links/pairing-code` | a code another instance can pair with (10 minutes, single use) |
| `POST`, `DELETE` | `/v1/links/{id}/use`, `/v1/links/{id}/refresh`, `/v1/links/{id}` | route tasks to a shared engine; re-check; unlink (both sides) |
| `PATCH` | `/v1/links/settings` | `share` and `gallery`; saved to `config/<profile>.local.yaml` |
| any | `/v1/links/{id}/r/v1/...` | a linked instance's conversation, artifact and job routes, through this server |
| `GET`, `POST` | `/v1/link/hello`, `/v1/link/pair`, `/v1/link/status`, `/v1/peer/engines/{name}/chat\|images` | between instances: pairing (its own proof), then the link's token. See [docs/linked-instances.md](docs/linked-instances.md) |
| `GET` | `/healthz` | liveness, no auth |

Errors are `application/problem+json` (RFC 9457) with a stable `code` and the `request_id`.
Every response carries `X-Request-ID`.

### Chatting

```bash
curl -N -X POST http://127.0.0.1:8700/v1/conversations/$CID/messages -H "content-type: application/json" -d '{"content": "Describe this", "attachments": ["<artifact id from an upload>"]}'
```

The response is `text/event-stream`. Each event has an increasing `id`:

| event | payload |
|---|---|
| `message` | a stored message: the user's, then the assistant's (`status: streaming`), then the assistant's final version |
| `delta` | `{message_id, content, reasoning}`: streamed text |
| `tool_call` | `{message_id, tool_call_id, name, arguments}`: the assistant called a tool |
| `job_update` | `{job}`: a generation job was queued, started, progressed, or finished |
| `artifact_created` | `{artifact, job_id}`: a new image is ready to fetch at `artifact.content_url` |
| `done` | `{turn_id, finish_reason, message_ids, usage}`; `finish_reason` is `cancelled` if stopped |
| `error` | a problem-details body; the assistant message is stored with `status: error` and any partial text. A chat stream that closes without a finish reason or `[DONE]` is `engine_interrupted`, not a short answer |

How a turn behaves:

- **Errors before the turn starts come back as plain HTTP errors:**
  - 404: no such conversation.
  - 409: a turn is already running in this conversation.
  - 422: bad attachments, or no backend has the needed capability. For example, an image sent to
    a model without vision is refused, never silently dropped.
  - 503: the engine is unreachable or still loading its model.
- **A turn runs independently of the HTTP connection.** If the client disconnects, the reply
  still completes and is stored. Re-attach with `GET .../turn/events`, or re-read `GET .../messages`.
- **What the model receives:** a system prompt listing the branch's artifacts by label (plus
  uploads no message has attached yet), the branch's last `context.max_messages` messages, and
  the newest `context.max_images` attached images as pixels, downscaled to
  `context.image_max_side`. Other branches are invisible to the model.

### Branches

A conversation is a tree. Every message has a `parent_id` (null for a first message), and
messages with the same parent are alternatives. A branch is the path from a first message to
a leaf. Clients that ignore branches keep working: without `parent_id`, a message continues the
**latest branch** (the one ending at the newest message), and `GET .../messages` shows it.

- **Continue any branch:** `POST .../messages` with `"parent_id": "<message id>"`, usually a
  branch's `leaf_id`. Continuing from a message that already has replies starts a new branch.
- **Edit & branch:** `"edit_of": "<one of your message ids>"` sends the new text as an
  alternative to that message. Its original branch stays as it was. Unknown ids, or `edit_of`
  pointing at a non-user message, are refused with 422 before anything is stored.
- **Retry:** `POST .../messages/{tool_message_id}/retry` runs the call behind that tool result
  again, with the same arguments and image options but a new seed. The new attempt is stored
  at once (`status: streaming`) beside the old result, filled in when the job finishes, and
  then the assistant replies on the new branch. It streams events like a message, starting
  with the attempt's `message` event. Errors before it starts: 404, 409 (a turn is running),
  422 (not a finished tool result).
- **Reading:** `GET .../messages?leaf_id=<message id>` shows the branch ending at any message;
  `?all_branches=true` returns every message so you can rebuild the tree. `GET .../branches`
  lists the branches, main first, each followed by the branches forked from it:

  | field | meaning |
  |---|---|
  | `leaf_id` | the branch's newest message |
  | `name`, `kind` | `main`; `message 2, edited` (`edit`); `retry of img-3` (`retry`) |
  | `fork_id`, `parent_leaf_id` | where it leaves its parent branch, and that branch's leaf |
  | `origin` | `from your first message`, `from assistant reply 1` |
  | `depth` | 0 for main, 1 for a branch of main, ... |
  | `message_count` | your messages plus assistant replies (tool results belong to their reply) |
  | `latest_image`, `latest`, `updated_at` | its newest image; whether it is the latest branch |

  At every fork, the oldest child continues its branch; `main` follows oldest children from
  the first message.
- **Labels** stay unique per conversation across branches: a retry's image is a new `img-N`.

### Tools and jobs

If the chat model supports tools and an image backend is configured, the model is offered
two tools. Both also take `negative_prompt`, `aspect_ratio`, `width`, `height`, `n` (up to 4),
`seed` and `transparent`.

- **`generate_image(prompt, ...)`** creates new images.
- **`edit_image(images, prompt, mask?, ...)`** edits, combines or restyles existing images,
  named by label (e.g. `["img-1", "img-3"]`):
  - **Inputs:** the first image is the one to change. Further images are references, for
    example a person, an object or a style. An optional `mask` limits the change to its white area.
  - **Lineage:** outputs are `edited` artifacts. The first input is recorded as `edit_source`,
    the others as `reference` in order, and the mask as `mask`.
  - **Output size:** follows the first input's shape, at the default image area.
  - **Large inputs** are downscaled before they're sent (`max_ref_pixels`).
  - **Labels** resolve only within the conversation. An unknown label comes back to the model
    with the list of labels that do exist.

- **Storage:** each call becomes a **job**. Its images are stored as `generated` (generation)
  or `edited` (editing) artifacts with
  new labels, attached to the tool's result message, and listed in the next system prompt.
- **Errors:** a failed call (bad arguments, a missing capability such as `transparent` on a
  backend without `rgba`, or an engine error) returns an error result to the model, which can
  explain it or retry. It never crashes the turn.
- **Tool rounds:** at most `orchestration.max_tool_iterations` rounds may call tools. The round
  after that is told `tool_choice: none`, so a turn always ends with an answer.
- **Queueing:** jobs queue per backend (`concurrency`, default 1 per GPU).
- **Reproducibility:** `n` images become `n` separate engine calls with seeds `seed, seed+1, ...`.
  Each output's seed is recorded, so any single image can be reproduced.
- **Cancelling:**
  - Cancelling a turn also cancels its running job on the engine.
  - `POST /v1/jobs/{id}/cancel` stops only the job. The turn continues, and the model is told
    the job was cancelled.
  - If a multi-image job fails or is cancelled midway, the images it already finished are kept
    and attached.
- **Restarts:** jobs left unfinished by a restart are marked `failed`.
- **User overrides:** a message may carry `image_options`, any of `aspect_ratio`, `width` and
  `height` (together; they win over `aspect_ratio`), `n`, `seed`, `steps`, `cfg_scale`,
  `negative_prompt`, `transparent`, `mask` (a `mask-N` label) and `backend` (an image backend
  name). Each one that is set overrides the model's arguments in every image call of that turn;
  `steps` and `cfg_scale` aren't tool arguments, so only the user sets them, and `mask` applies
  to `edit_image` calls. The result tells the model which values came from the user,
  and the user message stores them as `metadata.image_options`. An unknown or non-image
  `backend` is refused with 422 before the turn starts.

## Web UI

The server also serves a browser UI at `/`. It is plain HTML, CSS and ES modules in
`src/imaginer/web/`, with no build step, and it talks to `/v1` like any other client.
It follows the "Imaginer Main Interface" design, which is built on the Industry design system
(`web/ds/styles.css`, copied verbatim). Light and dark follow the OS setting.

- **Gallery:** every conversation is a tile showing its latest image, in a masonry that loads
  more as you scroll. Hover a tile for its label, title and age. A running turn shows its
  progress on the tile. Past the last tile, the prompt bar rises into the open space.
- **Prompt bar:** text, attachments (pick, paste or drop images) and quick-option chips. The
  chips open a tray for aspect ratio, image count, seed and image backend, and a second row for
  exact size, steps, CFG, a negative prompt and transparency, each showing the backend's own
  default. These are sent as `image_options`, "auto" leaves the choice to the model, and they
  persist in the browser. Only settings that differ from the default get a chip.
- **Search:** the server searches every conversation (archived ones too, if asked), page by page,
  with a bar tracking progress. Results show where they matched, highlighted. Esc clears, Enter
  opens the first result.
- **Viewer:** a tile opens on the conversation's latest image, with a reel of its other images.
  Images from other branches than the one shown carry a branch mark. You can retry, edit,
  mask or download from here. Focusing the bar opens the conversation, on the shown image's
  branch. A zoom bar fits, shows actual size or zooms to 400% (keys `+ − 0 1`; drag, arrows or
  Ctrl+scroll), with a minimap once the image outgrows the stage. Compare (`C`) pairs an edit
  with its source, as a slider or side by side.
- **Mask editor:** "Mask" on an image opens a brush to paint the area to change (B brush,
  E eraser, `[ ]` size, X invert, Ctrl+Z undo). The mask is uploaded at the image's size and
  goes into the bar beside it; the message sends it as `image_options.mask`.
- **Conversation:** streaming replies and collapsible reasoning. Each tool call shows as a row
  with its job state. A running job shows step progress, engine settings and a Cancel button.
  Outputs show their seed and size, plus Edit and Reference actions that put the image into the
  bar, and Retry. Edits list their lineage.
- **Branches:** Retry on an image runs its call again as a new attempt, and Edit & branch on any
  of your messages sends a new version of it. Both start a branch and nothing is deleted.
  "‹ 1 / 2 ›" switchers sit at each fork (on an edited message, and "attempt ‹ 1 / 2 ›" on a
  retried call). On wide screens a rail lists every branch with its message count, latest
  image and where it forked; the header's branch picker does the same on any screen. A reply
  continues the branch shown.
- **Engines:** the status line in any header opens the default engines, with Start and Log for
  local ones. Engine settings (`#/engines`) shows each backend's runtime, model files, endpoint,
  capabilities and startup log, and installs runtimes. See [Engines screen](#engines-screen).
- **Connection:** when the server can't be reached, a strip under the header counts down
  retries (1, 2, 4, 8, 16 s), running jobs dim to their last known step, and drafts are kept.
  Once back, running turns pick up from their last event.
- **Keyboard:** dialogs keep focus inside them and return it on close; menus move with the
  arrow keys, Home/End and letters; gallery tiles are one tab stop with arrow keys between them;
  toasts are announced to screen readers.
- **Access token:** if the server has `IMAGINER_SERVER__AUTH_TOKEN`, the UI asks for it once
  and keeps it in the browser's local storage.

## Development

```bash
uv run pytest
```

```bash
uv run ruff check
```

```bash
uv run ruff format --check
```

```bash
uv run mypy
```

- Tests need no GPU or engines. Adapters are tested against fake engines in `tests/fakes`,
  which speak the real wire formats over real sockets.
- Tests that talk to real engines are marked `integration` and are deselected by default:

```bash
IMAGINER_IT_LLAMA_URL=http://127.0.0.1:8080 IMAGINER_IT_SD_URL=http://127.0.0.1:1234 uv run pytest -m integration
```
- To add a migration after changing `src/imaginer/db/models.py`:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

- **CI** runs two jobs:
  - lint, format, mypy (strict), the migration drift check and the tests on Windows x64,
    Linux x64 and Linux arm64;
  - a packaging check that builds the wheel, installs it into a fresh venv, and migrates a
    database with it.
- New engine adapters: see [docs/adapters.md](docs/adapters.md) for the backend contract,
  including a design note for an SGLang Diffusion adapter.

## Engines

### Engines screen

A backend with a `launch` block can be started and stopped from the UI. The orchestrator then
runs the engine itself, listening on the backend's `base_url`, and stops it when it exits:

```yaml
backends:
  sdcpp:
    type: sdcpp
    base_url: http://127.0.0.1:1234
    launch:
      diffusion_model: D:/models/qwen_image_2.1-Q4_K.gguf
      vae: D:/models/qwen_image_2.1_vae_bf16.safetensors
      llm: D:/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
      llm_vision: D:/models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
      extra_args: []
      autostart: false     # start with the server
      # executable: ...    # default: the runtime installed under <data_root>/engines
```

`llamacpp` backends take `model` and `mmproj` instead. Without `launch`, a backend is managed
elsewhere and the screen only edits how this server reaches it. Changes made on the screen are
applied at once and saved to `config/<profile>.local.yaml`, which loads over the profile
(it's gitignored). The profile file itself is never rewritten, and API keys stay in the
environment. Starting an engine runs a local program, so keep the server on loopback or set a
token.

- **Changing a running engine's settings:** turns and image jobs already under way keep the
  connection they started with until they finish. Launch settings take effect at the next start.
- **Remote engines:** runtimes install on this machine, so an engine that isn't launched here
  and isn't on a loopback address shows no runtime and can't be installed from the screen.
  Update it on its own host. The same goes for a `linked` engine: its instance manages it.
- **Starting during an install:** an engine can't start while its runtime is being installed,
  and the runtime can't be installed while an engine using it runs.
- **Shutdown:** stopping the server cancels a running install and waits for it to clean up.

### llama-server (chat)

Imaginer includes the release provisioner and source builder adapted from
[Batch-Image-Interrogator](https://github.com/SmirkingKitsune/Batch-Image-Interrogator).
Install a local runtime explicitly:

```bash
uv run imaginer llama provision --profile nas
uv run imaginer llama provision --method source --accelerator cuda --cuda-arch 120 --jobs 2
uv run imaginer llama status
```

The default installation directory is `<data_root>/engines/llama_cpp`; override it with
`--cache-dir`. `auto` tries a matching official release, then a source build, then a CPU
release as a last resort. Any CPU fallback is reported in stderr and the JSON result.
Use `--method release` or `--method source` to require that method and accelerator.
Source builds require Git, CMake, a C++ compiler, and the selected accelerator's SDK.

`imaginer llama plan` previews steps, `imaginer llama check-update` checks upstream, and
`--tag <tag-or-ref>` pins an install. `--force` replaces an existing runtime, including
when updating `latest`. See [docs/llama-provisioning.md](docs/llama-provisioning.md) for
all options and the Python interface, which the [Engines screen](#engines-screen) also uses.

These commands install the executable and libraries. Start the returned executable with
the model arguments below, and configure the existing HTTP backend's `base_url` as usual,
or give the backend a `launch` block and start it from the Engines screen. `imaginer serve`
never downloads or builds engines, and launches one only if its `launch.autostart` is true.

The reference model is **Qwen3-VL-8B-Instruct**: vision and tool calling, and about 6 GB at Q4_K_M
plus its projector. Download both files from
[Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF):

- `Qwen3VL-8B-Instruct-Q4_K_M.gguf`: the model.
- `mmproj-Qwen3VL-8B-Instruct-F16.gguf`: the vision projector. Without it there's no image input.

sd-server uses the same two files as Qwen-Image-2.1's text encoder, so one copy on disk can
serve both.

Get llama.cpp from its [releases](https://github.com/ggml-org/llama.cpp/releases), tested with
`b11970`:

- **NAS (Linux x64, RTX 2000e Ada):** `llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`.
- **Windows with an RTX 5090:** `llama-<tag>-bin-win-cuda-13.4-x64.zip` plus
  `cudart-llama-bin-win-cuda-13.4-x64.zip`. The CUDA 12.4 build can't target `sm_120`.
- **DGX Spark (later):** `llama-<tag>-bin-ubuntu-cuda-13.4-arm64.tar.gz`.

Keep llama-server on `127.0.0.1`, next to the server that uses it. Another computer that needs
it links with that server and uses the engine it shares (see
[Linked instances](#linked-instances)), so the engine itself is never exposed.

**Linux (bash):**

```bash
llama-server -m ~/models/Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj ~/models/mmproj-Qwen3VL-8B-Instruct-F16.gguf --alias qwen3-vl-8b-instruct -c 32768 --host 127.0.0.1 --port 8080
```

**Windows (PowerShell):**

```powershell
.\llama-server.exe -m D:\models\Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj D:\models\mmproj-Qwen3VL-8B-Instruct-F16.gguf --alias qwen3-vl-8b-instruct -c 32768 --host 127.0.0.1 --port 8080
```

Notes:

- `--jinja` is on by default in current builds. Tool calling uses the template embedded in the GGUF.
- GPU layers are fitted automatically (`--fit`). Pass `-ngl` only to override that.
- If you start it with `--api-key-file` anyway, put the key in the env var your profile's
  `api_key_env` names (`IMAGINER_LLAMA_LOCAL_KEY` in `nas.yaml`).
- Then run `uv run imaginer backends check --profile <profile>`. For this model it should report
  `caps=chat,stream,tools,vision_in`.
- A missing `vision_in` means `--mmproj` wasn't loaded. A missing `tools` means the model's
  template has no tool support; override it with `--chat-template-file`.

### sd-server (images)

Provision the local engine with the same workflow as llama.cpp:

```bash
uv run imaginer sdcpp provision --profile windows-5090
uv run imaginer sdcpp provision --profile nas --method source --accelerator cuda --jobs 2
uv run imaginer sdcpp status --profile nas
```

The installation lives under `<data_root>/engines/stable_diffusion_cpp`. `auto` tries a
matching release, then a source build, then a CPU release if available. CPU fallback is
reported explicitly. `plan` previews steps, `check-update` checks upstream, and `--force`
replaces an existing runtime. Models are downloaded separately, and you launch the returned
`sd-server` executable with the arguments below. Source builds disable the bundled frontend.
See [docs/sdcpp-provisioning.md](docs/sdcpp-provisioning.md) for options and prerequisites.

The model is Qwen-Image-2.1 with three parts. Put them in one folder:

- **Diffusion model:** `qwen_image_2.1-Q4_K.gguf` (4.2 GB), or Q6_K/Q8_0 for more quality,
  from [leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF).
- **VAE:** `qwen_image_2.1_vae_bf16.safetensors` from
  [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae).
  Earlier Qwen-Image VAEs are not interchangeable with it.
- **Text encoder:** `Qwen3VL-8B-Instruct-Q4_K_M.gguf` plus `mmproj-Qwen3VL-8B-Instruct-F16.gguf`.
  These are the same files as the chat model. The mmproj is required for editing.

Get [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp/releases), tested with
`master-1024-77b72a2`:

- **Windows with an RTX 5090:** `sd-master-<sha>-bin-win-cuda12-x64.zip` plus
  `cudart-sd-bin-win-cu12-x64.zip` unzipped into the same folder. This build targets `sm_120`
  (CUDA 12.8.1), so no source build is needed.
- **Linux:** there is no prebuilt CUDA zip. Build with `-DSD_CUDA=ON`, see `docs/build.md`.

**Windows (PowerShell):**

```powershell
.\sd-server.exe --diffusion-model D:\models\qwen_image_2.1-Q4_K.gguf --vae D:\models\qwen_image_2.1_vae_bf16.safetensors --llm D:\models\Qwen3VL-8B-Instruct-Q4_K_M.gguf --llm_vision D:\models\mmproj-Qwen3VL-8B-Instruct-F16.gguf --listen-ip 127.0.0.1 --listen-port 1234
```

Notes:

- **Startup settings:** settings given on sd-server's command line (e.g. `--cfg-scale`) become
  its defaults, but each request from Imaginer sends the backend's `defaults` explicitly. Keep
  sampler settings in the profile.
- **Features to declare:** a model loaded from the command line reports every feature to the
  API, so set `declared_capabilities` to what the server can really do. Without `--llm_vision`,
  that's `[t2i, rgba]`: no edit.
- **Masks:** a mask is sent as one more reference image, with `mask_instruction` appended to the
  prompt. This is the "separate mask" input Qwen-Image-2.1 documents. How closely the model
  keeps the unmasked area is worth checking on your hardware; `tests/integration` runs it.
- **No auth of its own:** keep sd-server on `127.0.0.1`. Other computers use it through a
  [link](#linked-instances), with the link's token.
- **Check it:** `uv run imaginer backends check --profile windows-5090` should list `sdcpp`
  with `caps=edit,multi_ref,rgba,t2i`.

### Diffusers (in-process, optional)

The Hugging Face Diffusers `QwenImage21Pipeline` runs inside the orchestrator process. It is
the reference implementation for edit and RGBA output, and an alternative to sd-server when
you want the full-precision model. It needs about 30 GB of downloads on first use (the Hugging
Face cache), a lot of VRAM (`cpu_offload: true` trades speed for memory), and the optional extra:

```bash
uv sync --extra diffusers
```

- **Where torch comes from:** on Windows and Linux, this installs torch `2.14.1+cu130` from the
  PyTorch CUDA 13.0 index set in `pyproject.toml`. That build supports the RTX 5090 (`sm_120`),
  the NAS's Ada card, and Linux arm64. It needs an NVIDIA driver R580 or newer. For an older
  driver, point the `pytorch-cu130` index URL at `.../whl/cu128` and run `uv lock`.
- **Without uv:** `pip install ".[diffusers]"` gets torch from PyPI, which is **CPU-only on
  Windows**. Install a CUDA torch first, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
- **Enabling it:** set the profile's `diffusers` backend to `enabled: true`, and point `routing.t2i`
  and/or `routing.edit` at it. Nothing else changes: same tools, jobs, events and API.
- **Health:** `GET /v1/backends` reports missing packages, a CPU-only torch, or a model that failed
  to load. After a model-load failure, a usable runtime environment remains routable with
  status `ok` and the last error in its detail; the next generation or edit request retries
  loading. Health probes alone do not retry a failed load.
- **Loading:** the model loads on first use, or at startup with `preload: true`. Requests made
  during the load wait for it.
- **Its defaults** (`steps: 40`, `cfg_scale: 1.0`, 2048x2048) differ from sd.cpp's on purpose.
  Diffusers samples Qwen-Image-2.1 without guidance. `sampler`, `scheduler` and `flow_shift` don't
  apply and are ignored.
- **Seeds** use a CPU generator to improve reproducibility. Identical images across GPUs,
  library versions, or execution settings are not guaranteed; see the
  [Diffusers reproducibility guide](https://huggingface.co/docs/diffusers/main/using-diffusers/reusing_seeds).

### Verified engine facts (2026-10-28)

- **sd-server:** supports Qwen-Image-2.1 text-to-image, edit, and multiple references
  (`ref_images`, in order, up to 10). RGBA output is prompt-driven: there is no flag.
  Editing with a GGUF text encoder requires starting sd-server with
  `--llm_vision <mmproj-Qwen3VL-8B-Instruct-F16.gguf>`.
- **sd-server native API** (from `examples/server` source):
  - Each finished image carries `generation_info` with the seed actually used.
  - Step progress only exists while previews are on.
  - A generating job stops at the next checkpoint when cancelled.
  - A full queue answers `429`.
- **Windows build for the RTX 5090:** the official `sd-master-*-bin-win-cuda12-x64.zip` is built
  with CUDA 12.8.1 for `sm_120`. Unzip `cudart-sd-bin-win-cu12-x64.zip` next to it. No source build needed.
- **llama-server tool calling:** `--jinja` is on by default. The adapter reads
  `/props` → `chat_template_caps.supports_tools` and `modalities.vision` rather than assuming
  either. To run chat on the 5090, use the `win-cuda-13.4` build: CUDA 12.4 can't target `sm_120`.
