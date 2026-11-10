# Agents

An **agent** is a chat model plus a persona. The model is what llama-server loads: a GGUF, its
vision projector (`--mmproj`) and, optionally, a chat template (`--chat-template-file`), with a
context size and sampling. The persona is a display name, the system prompt it is told before
every conversation, a portrait, the tools it may call, and what the viewer's thought bubble
shows while it thinks.

The **agent library** (`#/agents` in the UI, `GET /v1/agents` in the API) lists them like the
model library lists image models: a tile per agent with its portrait as the cover.

## Which agent answers

A message goes to:

1. the agent it names (`MessageCreate.agent`);
2. else the conversation's agent (`metadata.agent`, set by its first agent turn);
3. else the **default agent**: the one the default chat engine is set to run (its `agent`),
   else the first agent made for that engine.

The agent's engine answers. It is told the agent's system prompt (or the built-in one), is
offered only the tools the agent has on, and gets the agent's sampling with each request, over
the engine's own `sampling`. Each assistant message records its agent (`Message.agent`), so the
conversation shows the agent's portrait and name beside each reply, and the agent card counts
its turns and images.

A message that names a `backend` but no agent runs as it did before agents, without a persona.
Raw mode runs no LLM, so it has no agent.

## Engines

- **A llama-server launched here** runs the agent its `agent` setting names (saved to the
  profile's `.local.yaml`, like an sd-server's `model_card`). Its config as it runs has the
  agent's model, vision projector, chat template and context size over its own. **Use on**
  (`POST /v1/agents/{id}/use`) points the engine at the agent and restarts it if it is running;
  it is refused while a turn is using the engine. A turn as an agent whose engine runs another
  agent's model is refused with 409, so a persona never answers with the wrong model.
- **An engine started elsewhere** (a remote llama-server, or one a linked instance shares)
  runs its own model. Its agents carry no files of their own; their persona applies to whatever
  it runs. Several agents can share it; "Make it the default" sets which one new prompts go to.

**Moving in.** At startup every chat engine without an agent gets one: made from the model
files of a llama-server launched here (which is then pointed at it), or named after the model
an engine started elsewhere reports.

## Portraits

An agent's portrait shows in the viewer while it thinks and beside its replies. It is one of:

- **From Hugging Face:** the avatar of whoever uploaded its GGUF repo (the repo's owner, an
  organization or a user). It is fetched when the agent is made from a Hub reference or a file in
  the Hugging Face cache, and becomes the portrait while it has no other. "Refresh from the Hub"
  fetches it again.
- **Self-portrait:** the agent paints one with the `set_portrait` tool. It writes its own prompt;
  the image is painted square on the image engine, stays in its conversation like any other image,
  and a square copy becomes the agent's portrait as its next version (v1, v2…). Any earlier one can
  be shown again. The tool's guidance asks the agent to ask before changing its portrait, unless the
  user asked for it. **Paint another** (`POST /v1/agents/{id}/portrait/paint`) starts a conversation
  in which the agent is asked to paint one and is offered only `set_portrait`.
- **Upload:** any image, cropped to its centre square (`POST /v1/agents/{id}/portrait/upload`), or
  an image from a conversation (`PUT /v1/agents/{id}/portrait` with `artifact_id`).

Without any, the UI shows the agent's monogram: the first letter of its name.

Portraits are kept as square WebP files under `<data_root>/agent_portraits/<agent id>/`.

## The viewer

A prompt sent from the gallery opens the conversation's viewer:

- **No images yet:** the agent's portrait stands in the middle of the stage. While it thinks, a
  thought bubble above it shows how long it has thought, a status line ("Planning the
  composition", "Looking at img-3") and, unless the agent shows status only, the tail of its
  reasoning as it streams. When it answers, the bubble becomes a speech bubble with its reply and
  the call it makes.
- **With images:** the same bubbles sit in the stage's corner, beside a small portrait. A finished
  reply stays until it is closed; Read more opens the conversation.
- **While the image engine paints:** the bubbles clear. The stage shows the image's steps loading,
  in its shape, with the job's progress, the engine and seed, and Cancel; the side column describes
  the image on its way (its prompt, size, steps), and the reel has a tile for it and for each one
  still queued.

## API

| Method | Path | |
|---|---|---|
| `GET` | `/v1/agents` | every agent, the chat engines, and the default agent |
| `POST` | `/v1/agents` | a new agent: `path` (a GGUF here, with `mmproj`), `ref` (a Hub reference), or neither (whatever `engine` runs) |
| `GET`, `PATCH`, `DELETE` | `/v1/agents/{id}` | one agent: name, engine, files, system prompt, viewer, tools, sampling, context, notes, tags, pin |
| `POST` | `/v1/agents/{id}/use`, `/stop` | run its model on its engine (restarting it), or stop that engine |
| `PUT` | `/v1/agents/{id}/portrait` | show one of its portraits (`key`), an image from a conversation (`artifact_id`), the Hub avatar fetched again (`refresh`), or none (`key: none`) |
| `POST` | `/v1/agents/{id}/portrait/upload`, `/portrait/paint` | upload a portrait; have it paint a self-portrait |
| `GET` | `/v1/agents/{id}/portraits/{key}` | a portrait (`hub`, `upload`, `self-<n>`) as WebP |

`POST /v1/hub/downloads` takes `agent` and `key` (`model` or `mmproj`) to download a file from the
Hub into an agent.

Sampling an agent may set: `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`,
`presence_penalty`, `max_tokens`. A `null` value removes one.
