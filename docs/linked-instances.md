# Linked instances

An Imaginer instance is **standalone** by default: one computer runs the server and its
engines, and keeps every conversation. Linking is for when two or more instances share a
network and one of them has a GPU the others can use:

- **Offloading:** an instance keeps its conversations and sends some tasks (chat, t2i, edit) to
  an engine a linked instance **shares**. The whole model runs on the sharing instance; there is
  no layer splitting.
- **One gallery:** each instance lists its linked instances' conversations next to its own.
  They stay at their origin: opening or continuing one goes through this server to that
  instance, and its turns run there, with its routing and engines.

There is no controller. A link is a symmetric pairing between two instances, and nothing is
transitive: an instance serves only its own conversations and its own (non-linked) engines,
so a request never hops on to a third instance.

## Configuration

```yaml
server:
  host: 0.0.0.0             # reachable from the network
links:
  enabled: true
  name: nas                 # shown to linked instances; default: the hostname
  advertise_url: null       # how they reach this one; default http://<hostname>.local:<port>
  discovery: true           # advertise and browse over mDNS (_imaginer._tcp)
  poll_interval_s: 30       # how often each link's status is refreshed
  start_timeout_s: 300      # how long a stopped shared engine may take to load
  gallery: full             # off | read | full: what linked instances may do with conversations here
  share:                    # engines linked instances may use
    llama: {remote_start: true}
```

A backend that uses a shared engine:

```yaml
backends:
  llama_nas:
    type: linked
    link: nas        # this instance's name for the link
    engine: llama    # the backend's name on the linked instance
    kind: llm        # llm or image
    concurrency: 1   # image jobs sent at once (the sharing side queues them too)
    declared_capabilities: null  # optional upper bound, like any backend
```

Rules the server enforces at startup:

- Linking beyond this machine needs `IMAGINER_SERVER__AUTH_TOKEN`. Only two instances that are
  both on loopback (development) may link without one.
- With `server.host` on loopback, `links.advertise_url` must be set (a reverse proxy).
- `links.share` can only name local backends, never `linked` ones. `null` un-shares an engine
  the profile shares (the Links screen writes that to `config/<profile>.local.yaml`).

## Pairing

1. **Responder** (the instance being linked to) mints a code: **Allow a link** on its Links
   screen, `POST /v1/links/pairing-code`, or `imaginer link code` on a headless host. A code
   is 8 characters of Crockford base32 (about 40 bits), valid for 10 minutes, single use, and
   gone after 5 wrong tries. A new code cancels the previous one. Only its SHA-256 is stored.
2. **Requester** asks `GET /v1/link/hello` (no token) for the responder's instance id, name,
   link API version and a fresh challenge (kept for 2 minutes).
3. **Requester** sends `POST /v1/link/pair` (no token) with
   `proof = HMAC-SHA256(key=sha256(code), "imaginer-pair/1\nchallenge\nrequester id\nresponder id")`,
   its own id, name, URL and API version, and a random token the responder will call it with.
   The code itself never crosses the network, and a proof can't be replayed or reused.
4. **Responder** checks the proof in constant time, burns the code, stores the link and answers
   with a token the requester will call it with. Each side stores the token it was given (to
   call the other) and only the SHA-256 of the token it issued.
5. **Responder** confirms it can reach the requester back, at its advertised URL or else at
   the address the request came from.

Each address may fail pairing 10 times in 10 minutes. Pairing again with an instance that is
already linked replaces the tokens. Unlinking (`DELETE /v1/links/{id}`) tells the other side
to forget the link too (`POST /v1/link/revoke`), if it can be reached.

Identity is the instance id, a UUIDv7 made on first start and kept in the database. An
instance found at a new address (by mDNS, or by the address its requests come from) is moved
there only if the same instance answers there with the same token.

## The peer API

Everything below needs the link's token, and answers 404 while linking is off.

| Method | Path | |
|---|---|---|
| `GET` | `/v1/link/status` | `PeerStatus`: id, name, API version, gallery mode, and every shared engine |
| `POST` | `/v1/link/revoke` | the caller unlinked: forget it |
| `POST` | `/v1/peer/engines/{name}/chat` | one chat completion, as server-sent events |
| `POST` | `/v1/peer/engines/{name}/images` | one image (generate or edit), as server-sent events |

With `links.gallery` set to `full` (or `read`, GET only), a linked instance may also use the
conversation, message, artifact and job routes, and `GET /v1/backends`. It may never use the
Engines or Links routes, the merged gallery, or the proxy, so nothing is transitive.

### Shared engines and fit

Each shared engine in `PeerStatus` carries its model, health, capabilities, image defaults,
queue depth and an `availability`:

| `state` | Meaning |
|---|---|
| `ready` | Healthy: the model is loaded. |
| `cold` | Stopped (or loading), and `remote_start` allows starting it: it loads on first use. |
| `unfit` | It would not fit: `need_mb` is more than free VRAM minus 1 GiB. |
| `unavailable` | Stopped and can't be started for a linked instance: no `remote_start`, no launch block, no runtime, or missing model files. |

`need_mb` is an estimate from the launch files' sizes: llama weights × 1.2 (KV cache and
buffers), sd.cpp weights + 1.5 GiB (activations). VRAM comes from `nvidia-smi` (the first GPU),
cached for 5 seconds. Without either, the fit is unknown and a startable engine is `cold`.

A stopped engine has no capabilities to report, but the calling side routes by them, so a
`cold` engine reports the ones it will have once loaded: from its launch block (llama.cpp:
vision only with `mmproj`, tool calling assumed), narrowed by `declared_capabilities`. The
loaded engine is still checked before each request runs on it.

### Streams

`chat` takes `{messages, tools, tool_choice, params}` and sends:

- `status` `{state: "loading"|"ready"}` while a cold engine starts,
- `delta` `{content, reasoning, tool_calls[], finish_reason, usage}` for each chunk,
- then `done`, or `error` with problem details (`status`, `code`, `title`, `detail`).

`images` takes `{op: generate|edit, prompt, seed, negative_prompt, width, height,
aspect_ratio, alpha, params, images: [{data_b64, mime, label, width, height}], mask}` and sends
`status` (`loading`, `queued`, `running`), `submitted` `{request_id}`, `progress` `{step,
total_steps}`, `result` (the image as base64, with its size, alpha, seed and resolved
params), then `done` or `error`.

Image requests wait at the same gate as the sharing instance's own jobs (`concurrency`), so
a GPU never runs more than it is set to, whoever asked. When the caller disconnects, the
sharing side cancels the work, and with it the engine's job.

On the calling side the `linked` adapter maps errors back to the same kinds
(`engine_unavailable`, `engine_timeout`, `capability_unavailable`, `invalid_request`, else
`engine_error`), by code rather than by message. A stream is retried only until its response
headers arrive, never after it has started, so a reply can't be duplicated. Routing never
falls back silently: an unreachable link is `engine_unavailable`, like any engine that is down.

## The gallery and remote conversations

`GET /v1/gallery` merges this instance's conversations with each linked instance's
`/v1/conversations` (asked in parallel, 3 seconds each). The cursor holds every source's
position. A source that doesn't answer is reported in `sources` and skipped for the rest of
that listing. Each item has an `origin`: `link_id`, `instance_id`, `name`, and `base_path`.

`/v1/links/{id}/r/<path>` passes a request on to the linked instance with the link's token
and streams the answer back (server-sent events, `ETag`, `Last-Event-ID` included). Only
`v1/conversations/...`, `v1/artifacts/...`, `v1/jobs/...` and `GET v1/backends` pass, and
the linked instance still applies its own `links.gallery`. The browser only ever holds its
own server's token. The UI's routes for a remote conversation start with `#/l/<link id>`.

## Discovery

With `links.discovery`, an instance advertises `_imaginer._tcp.local.` with TXT records `id`,
`name`, `v` (link API version) and `url`, and browses for the others (`GET
/v1/links/discovered`). Discovery only lists instances; linking always takes a code. Where
multicast doesn't work (containers, VPNs), link by address instead.

## Security model

- Tokens are random (256 bits), per link and per direction. Each side keeps only the hash of
  the token it issued, and compares secrets in constant time.
- After pairing, tokens travel as bearer headers over HTTP, like the server token. Link on a
  trusted network, or put a TLS reverse proxy in front and set `links.advertise_url` to its
  `https://` address.
- Engines stay on `127.0.0.1`. Linked instances reach them only through the server, and only
  those listed in `links.share`.
- The outbound tokens (to call each peer) are stored in the database under `data_root`, never
  in YAML.

## Command line

```bash
uv run imaginer link code --profile nas
```

```bash
uv run imaginer link add http://nas.local:8700 --code K7QM-2XPD --name nas --profile windows-5090
```

```bash
uv run imaginer link list --profile windows-5090
```

```bash
uv run imaginer link remove nas --profile windows-5090
```

These work on the database directly, so they also work while the server runs: it picks up
added or removed links on its next poll.
