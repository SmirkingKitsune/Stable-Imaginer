# The artifact library

The **artifact library** (`#/artifacts` in the UI, `GET /v1/artifact-library` in the API) lists
every image uploaded or made in any conversation, on this instance and on linked ones, newest
first. Any of them can go into a new prompt.

An artifact still belongs to exactly one conversation. Using a library image in another
conversation **copies** it there: it becomes an upload of that conversation, with a label of its
own (`img-N`, or `mask-N` for a mask), so `@img-N` mentions, `edit_image` and masked edits treat
it like any other image. The copy's metadata says where it came from (`imported_from`: the
instance, conversation, label, prompt and how it was made), and the chat model is told so in its
list of the conversation's images.

## What is listed

- **Each distinct image once.** Copies of the same bytes (an image used in three conversations)
  are one entry: the earliest copy, with `copies` saying how many there are. When that copy's
  conversation is deleted, the next one stands in.
- **Images by default; masks when asked** (`kind=mask`, or `kind=all` for both).
- **Filters:** where it came from (`source`: `upload`, `generated`, `edited`, repeatable),
  `pinned`, one instance (`origin`: `local` or a link id), and a search (`q`) over the prompt,
  file name, label, conversation title, and a pin's name and notes.
- **How it was made**, for generated and edited images: the backend, model and model card, seed,
  and the settings the engine ran (`generation.resolved`: size, steps, cfg, sampler, negative
  prompt, LoRAs), from the job that made it.
- Archived conversations' images are listed too.

## Pins

**Pinning** an image keeps a copy of its bytes in `<data_root>/artifact_library/`, with a
snapshot of where it came from. While a conversation still has the image, the library lists the
live image with its pin. Once none does (its conversation was deleted, or it came from a linked
instance), the pin itself is listed (`live: false`) and serves the bytes. Unpinning such an image
deletes it for good, so the UI asks first.

A pin has a name, notes and tags of its own, searched with the rest. Pinning the same bytes
twice gives the same pin back.

The viewer's **Pin** keeps the image shown. In the library, pin from an image's details or from
the bar that appears when tiles are selected.

## Using an image

- **From the prompt bar:** the paperclip opens a small menu: **Upload from this device** (a new
  file, which joins the library once the message is sent, like any image of a conversation), or
  **From the artifact library**, which opens the library as a picker. Dropping or pasting files
  into the text still uploads them. Pick up to as many images as the message has room for (16, or
  what the edit model takes in raw mode). They show in the bar as `lib-…` until the message is sent; then each is copied into the
  conversation and the text's `@lib-…` becomes its new label.
- **From the library:** **Use in next prompt** (one image's details, or several selected) puts
  them in the prompt of the place you came from (a conversation, or the gallery for a new one)
  and goes back there. **Edit in next prompt** also starts the text as `Edit @…:`.

An image the conversation already has is attached as itself, not copied again; the server also
returns the existing copy when the same bytes are imported twice (`200` instead of `201`).

## Linked instances

The listing asks every linked instance for its own images (`GET /v1/artifact-library/local`, 3
seconds each, in parallel), as the gallery does for conversations, and merges the pages by time
(artifact ids are UUIDv7). A remote image this instance already has, live or pinned, is left
out, so a page can come back short, or even empty, while `next_cursor` says there is more.

Each instance's status is in `sources`:

| Status | Meaning |
|---|---|
| `ok` | listed |
| `unreachable` | didn't answer; reported at once next time and asked again in the background |
| `unsupported` | an Imaginer from before the artifact library: update it to see its images |
| `error` | it answered, but refused (its `links.gallery` is `off`) or sent something unexpected |

Using or pinning a remote image fetches its bytes through the link (checked against the sha256
it was listed with, at most `storage.max_upload_bytes`). Importing into a conversation that
lives on a linked instance uploads the image there through its own upload route; that needs its
`links.gallery` to be `full`.

A linked instance sees only this instance's own images: its live ones, and pins of its own
images. Pins of images that came from somewhere else are never passed on, and a pin's name and
notes stay here.

## API

| Method | Path | |
|---|---|---|
| `GET` | `/v1/artifact-library?limit=&cursor=&kind=&source=&pinned=&origin=&q=` | every image here and on linked instances; each item has a `ref` (send it back to pin or import), an `origin` (`base_path` + `content_url` loads it), `generation`, `copies`, `pin`; each page has `sources` |
| `POST` | `/v1/artifact-library/pins` | `{ref, name?, notes?, tags?}`: keep it; `201`, or `200` with the pin these bytes already have |
| `GET`, `PATCH`, `DELETE` | `/v1/artifact-library/pins/{id}` | a pinned image as listed; its name, notes, tags; unpin (deletes the kept copy) |
| `POST` | `/v1/artifact-library/import` | `{conversation_id, link_id?, ref}`: copy it into a conversation (here, or on the linked instance `link_id`); the artifact it became; `201`, or `200` when the conversation has it already |
| `GET` | `/v1/artifact-library/local`, `/local/artifacts/{id}`, `/local/pins/{id}` | this instance's own images (what linked instances list), as far as `links.gallery` allows |
| `GET` | `/v1/artifact-library/pins/{id}/content` | a pin's bytes (`ETag` = sha256, cached for good); also through `/v1/links/{id}/r/` |

The first four are the user's only; linked instances call the last two, under `links.gallery`.
