# Hugging Face models

The Engines screen finds models on the Hugging Face Hub, works out which files in a repo each
engine setting can use, and downloads them. It also finds copies you already have, in the
models folder or the Hugging Face cache. Press **Hugging Face** on an engine's model files.
You can also paste a reference into any path field: a Hub URL (`https://huggingface.co/...`,
`hf.co/...`), an `hf://` path, or llama.cpp's `owner/name:QUANT`. The repo opens with that
file or quant picked.

## How repos are read

Hub repos come in a few standard shapes. The server reads the repo's file list (and
`model_index.json` for a Diffusers pipeline) and sorts what it finds:

| Shape | Looks like | Used by |
|---|---|---|
| **GGUF quants** | `<name>-Q4_K_M.gguf`, `-UD-Q4_K_XL`, `-BF16`… side by side, or one folder per quant with `-00001-of-00003` shards; `mmproj-*.gguf` vision projectors | llama-server (`-m`, `--mmproj`); sd-server (`--diffusion-model` for image repos, `--llm` and `--llm_vision` for LLM repos) |
| **Diffusers pipeline** | `model_index.json`, and `transformer/`, `vae/`, `text_encoder/`… with sharded safetensors and a `*.safetensors.index.json` | the diffusers backend, as a whole (`model_id`); sd-server, one part per setting, through its index |
| **ComfyUI repackage** | single files under `diffusion_models/`, `vae/`, `text_encoders/` (sometimes inside `split_files/`) | sd-server |
| **transformers checkpoint** | `config.json`, `model-0000N-of-0000M.safetensors`, `model.safetensors.index.json` | sd-server's `--llm`, through its index, for the text encoders it supports; llama.cpp needs a GGUF conversion, which **Quantized versions of this model** finds |
| **Loose safetensors** | anything else | sorted by folder and name (`vae`, `text_encoder`, `lora`…) |
| **LoRA adapter** | a repo the Hub tags `lora` or `base_model:adapter:…`; its weights are LoRAs, whatever they are called, unless a folder (`vae/`…) names another part | the engine's LoRA folder: sd-server's `--lora-model-dir`, Diffusers' `lora_dir` (`.safetensors` only) |

**LoRAs** (single files under `loras/`, named `*lora*`, or in an adapter repo) get their own
**LoRAs** section for an sd-server launched here and for a Diffusers backend (which loads only
`.safetensors` LoRAs). **Install** puts one in the engine's LoRA folder (`--lora-model-dir`,
`lora_dir`) as `<folder>/<owner>/<name>/<path in the repo>`, where the engine finds it without
a restart. An engine with no LoRA folder yet gets `<models_dir>/_loras` with its first LoRA
(no Hub namespace starts with `_`); an sd-server then needs a restart to load it. A LoRA
already on this machine, in the models folder or the HF cache, is copied in rather than
downloaded again (the dialog says so). Apply LoRAs with a preset or from the prompt bar.

`imatrix` files, ControlNets, pickled weights (`.bin`, `.ckpt`), and LoRAs for any other
engine are listed as files that no engine setting loads. When a pipeline's weights are only published as a Diffusers
variant (`*.fp16.safetensors`), the dialog says so, because the diffusers backend doesn't
request a variant by name.

For each setting the dialog suggests one file. It picks, in this order:

1. the quant the reference asked for (`repo:Q6_K`) or the file a URL pointed at;
2. the quant the setting uses now, so switching repos keeps your quality level;
3. a sensible default: Q4_K_M, then nearby K-quants, then larger ones (F16 for projectors);
   the plainest name wins a tie (official over `non_official/`, base over `turbo`).

Sharded safetensors are loaded through their `*.safetensors.index.json`. sd-server has read
these since stable-diffusion.cpp #1769 (July 2026). It also converts Qwen-Image 2.1's Diffusers
VAE and transformers Qwen3-VL tensor names, so a Diffusers download of Qwen-Image 2.1 can fill
`--diffusion-model`, `--vae` and `--llm`. The sd.cpp docs themselves use the ComfyUI and GGUF
files.

## Model cards

A model card's slots (see [models.md](models.md)) open the same dialog for one slot at a time:
its family's curated sources first, then any repo. A download lands in the card's slot, and an
engine that runs a card downloads into the card. For sd-server, a repo's files are matched
to every slot a family can have:

- **`--model`:** a whole checkpoint, at the repo's top level or in `checkpoints/`.
- **`--diffusion-model` and `--uncond-diffusion-model`:** Ideogram 4's
  `unconditional_transformer/` goes in the second.
- **`--vae` and `--tae`:** a tiny autoencoder (`taesd*`, `taef1`…) only in `--tae`.
- **`--clip_l`, `--clip_g`, `--t5xxl` and `--llm`:** by the Diffusers class of the folder
  (`CLIPTextModel`, `text_encoder_2`'s OpenCLIP bigG, `T5EncoderModel`), else by name
  (`clip_l`, `clip_g`, `t5xxl`, `umt5`, `flan-t5`). Anything else, such as Qwen3, Mistral
  or Gemma, is an LLM.
- **`--embeddings-connectors`:** `*connector*` files.
- **`--tokenizer`:** `tokenizer.json`.

A named file is suggested over a Diffusers part of the same thing (`ae.safetensors` over
`vae/diffusion_pytorch_model.safetensors`).

## Model file checks

The Engines screen checks model files against what each engine can load:

- **llama-server:** GGUF only. A split GGUF must be given by its first shard
  (`-00001-of-0000N`); llama.cpp finds the rest.
- **sd-server:** GGUF or safetensors, or a safetensors index: a `*.safetensors.index.json`,
  or a folder holding `model.safetensors.index.json`. A single shard of a sharded checkpoint
  would load part of a model, so it is flagged with the index to use instead. `--tokenizer`
  takes a `tokenizer.json`. Which slots must be filled is the model's family's: FLUX.1 needs
  CLIP-L, T5-XXL and a VAE (or a tiny autoencoder), SDXL only its checkpoint.

## Where files go, and copies already on disk

Downloads go to the **models folder**, `hub.models_dir` (default `<data_root>/models`), as
`<models_dir>/<owner>/<name>/<path in the repo>`. That's the same tree as
`hf download --local-dir`. Change it from the dialog's footer; it is saved to
`config/<profile>.local.yaml`.

Before offering a download, the dialog looks for a complete copy, every file at the size the Hub
lists, in:

1. the models folder's own tree;
2. Hugging Face cache trees inside the models folder: `<models_dir>/models--<owner>--<name>/`,
   or one per owner, `<models_dir>/<owner>/models--<owner>--<name>/`, as
   `hf download --cache-dir` leaves them;
3. the Hugging Face cache itself (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else
   `~/.cache/huggingface/hub`), where `hf download`, Diffusers and transformers put what they
   fetch. Turn this off with `hub.use_hf_cache: false`.

A copy found there is **used in place**: **Use** points the setting at it. **In use** marks the
copy a setting already points at, including through a mapped drive or a cache symlink.

## Downloads

- They run on the server and continue when the dialog closes. The Engines screen shows the
  progress next to the setting being downloaded.
- Files are written as `<name>.part` and renamed into place when complete. An engine is never
  pointed at a partial file.
- An interrupted download (cancelled, a dropped connection, or a server restart) resumes from
  its `.part` with a range request. Network errors are retried three times this way.
- LFS files are checked against the SHA-256 the Hub lists. A file that doesn't match is
  deleted and downloaded again.
- A LoRA install copies a complete copy found on this machine instead of downloading it,
  checked against the Hub's SHA-256 the same way. A copy that doesn't match, or can't be read,
  is downloaded instead, and the download says why.
- All the files of one download (shards, pipeline parts) come from the same commit.
- Disk space is checked before starting.
- When a download finishes, it sets the engine setting it was started for. This is the same
  edit you'd make on the Engines screen, saved to `config/<profile>.local.yaml`. Settings
  are not changed by a download that fails or is cancelled.

## Tokens, gated repos and mirrors

The server uses the same credentials as the `hf` CLI, so a machine that is already logged in
needs nothing else:

- **Token:** `$HF_TOKEN` (or the env var `hub.token_env` names), else the file `hf auth login`
  writes (`$HF_TOKEN_PATH`, else `$HF_HOME/token`). It is read for each request, so logging
  in takes effect without a restart. It is sent to the Hub only (not to its CDN), and never
  logged or returned by the API.
- **Gated repos** (marked GATED) need their terms accepted on the Hub, by the account whose
  token the server uses.
- **Mirrors:** `hub.endpoint`, else `$HF_ENDPOINT`. Links from a mirror can be pasted too.

```yaml
hub:
  enabled: true              # false: no Hub access at all
  # endpoint: https://hf-mirror.com
  # token_env: HF_TOKEN
  # models_dir: N:/models    # default: <data_root>/models
  # use_hf_cache: true
```

## API

| Route | |
|---|---|
| `GET /v1/hub` | Endpoint, whether a token is set (never the token), the models folder and HF cache |
| `PATCH /v1/hub` | `{"models_dir": ...}` |
| `GET /v1/hub/search` | `q`, `format` (`gguf`/`diffusers`/`safetensors`/`any`), `task` (a pipeline tag), `sort` (`downloads`/`trending`/`likes`/`updated`), `base_model` with `relation=quantized` for a model's GGUF versions |
| `GET /v1/hub/repo?ref=...&engine=...` | The repo's layout, local copies, and each of the engine's settings with its candidates and suggestion |
| `POST /v1/hub/downloads` | `{"repo", "variant", "engine", "key"}`; without `variant`, the Diffusers pipeline, for a diffusers backend's `model_id` |
| `GET /v1/hub/downloads`, `POST /v1/hub/downloads/{id}/cancel` | |
| `GET /v1/hub/events` | `download` events (server-sent) |

These routes are for the user only, like the Engines screen's: linked instances can't reach
them.
