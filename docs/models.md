# Image models: the model library

sd-server runs many image model families. Some are a single checkpoint (SD 1.x, SDXL). Some
are a diffusion model plus CLIP and T5 text encoders and a VAE (FLUX.1, SD3.5). Some are a
diffusion model with an LLM text encoder (FLUX.2, Z-Image, Qwen Image). A few need a
tokenizer (MOSS, Lens), and pixel-space ones need no VAE at all (Chroma1-Radiance, Iris).

The **model library** (`#/models` in the UI, `GET /v1/models` in the API) keeps a **model card**
for each model you import. Each card holds:

- **its family:** one of sd-server's recipes (below);
- **its components:** the file in each of the family's slots (`--diffusion-model`, `--vae`,
  `--clip_l`, `--t5xxl`, `--llm`…);
- **its launch arguments:** `--vae-format`, `--model-args`, and more arguments such as
  `--offload-to-cpu`;
- **its own defaults and presets:** size, steps, CFG, distilled guidance, sampler,
  scheduler, sigmas, negative prompt. What suits SDXL is wrong for FLUX, so these belong
  to the model, not the engine;
- **a note on how to prompt it,** which the chat model is told: tags for SD 1.x/SDXL,
  structured JSON for Ideogram 4, plain description for most others;
- **what you add:** notes, tags, a pin and a cover. Usage comes from the images each job
  records.

## Families

A family is one of stable-diffusion.cpp's **sd-server recipes** (`examples/server/recipes`).
A recipe says:

- which component slots the family loads, and which are required (or one of a group, like
  a VAE or a tiny autoencoder);
- its generation defaults;
- the multiple its sizes snap to;
- its features, such as reference images for editing and distilled guidance.

Stable-Imaginer bundles a snapshot (`src/imaginer/image_models/data/sdcpp_recipes.json`). A
running sd-server's own recipes (`GET /sdcpp/v1/recipes`) replace it, so slots and defaults
match the build in use. Refresh the snapshot with:

```bash
uv run python scripts/sync_sdcpp_recipes.py            # or --ref <commit>
```

Our notes on each family are in `src/imaginer/image_models/data/families.yaml`:

- how the library groups it;
- which LoRAs fit it;
- how to prompt it;
- its native size, where that is larger than the recipe's (Qwen-Image-2.1 is 2K);
- its transparent-output and mask instructions;
- the `--model-args` its sd.cpp doc recommends, such as `chroma_use_dit_mask=false`;
- the name, repo and GGUF-architecture hints used to detect it;
- **curated Hugging Face sources** for each slot, taken from the sd.cpp docs.

Edit, control and decoder recipes (Qwen-Image-Edit, Kontext, ControlNets, PiD…) are hidden from
the library. Video recipes are not included.

What a family can do follows its recipe:

- edit and multi-reference need its `ref_images` feature (Qwen-Image-2.1, FLUX.2, FLUX.3,
  Fibo, P-H3…);
- transparent output needs a prompt template (Qwen-Image-2.1, Ming-Image);
- an edit with a mask needs a mask instruction (Qwen-Image-2.1).

An engine running a family that lacks one of these doesn't offer it.

## Importing a model

**Import a model** takes three steps:

1. **Source.** Drop a `.gguf` or `.safetensors` on the dialog, browse for one, paste its path,
   or search Hugging Face (the same browser as the Engines screen; the file you pick
   downloads into the new card). A browser tells the server a dropped file's name and size,
   not its folder, so the server looks for it (`POST /v1/models/locate`) in the download
   folder, the Hugging Face cache, the folders of every card's files and the folder above
   each, and Downloads. If it isn't there, paste its full path.
2. **Family.** The guesses, best first, each with its confidence and the evidence for it.
   Any other family can be picked instead.
3. **Files.** The family drawn as its pipeline (prompt, text encoders, model, decoder,
   image). Each file it needs is matched to one already here: a file a card of the same
   family group uses (FLUX.1 dev and schnell share CLIP-L, T5 and the VAE) is taken as is,
   anything else is offered for you to pick, with downloads from its curated sources. Leave
   one empty to pick it later; the card shows what it still needs. Downloads start once the
   card is made and land in its slots.

The family is detected from the strongest evidence first:

1. **the file's tensor names**: these tell SD 1.x, SD 2.x and SDXL checkpoints apart, SD3's
   joint blocks (with their depth), and FLUX.1 with or without a guidance input, or Chroma;
2. **the Hub repo it came from**, or the `base_model` its card names;
3. **the Diffusers pipeline class**;
4. **the names** of the file and its folders;
5. **GGUF `general.architecture`**.

Detection is a guess you confirm, so the card shows why it guessed. On the card, a setup plate
counts the files it has against those its family needs, with the one next action (use the
copy on disk, download it, or find it). A missing file opens into the ways to get one. Fill
each slot in any of these ways:

- **a path:** type it, or paste a Hugging Face URL;
- **Reuse:** a file another card already uses. A FLUX VAE, a T5 encoder or a Qwen3 text
  encoder is shared rather than downloaded again. The library's **Files on disk** list
  shows every component file and the cards that use it;
- **a curated source:** for example `comfyanonymous/flux_text_encoders/clip_l.safetensors`
  for FLUX.1's `--clip_l`;
- **Hugging Face:** search the Hub for that slot. A download lands in the card's slot.

On the Hub, the dialog matches a repo's files to the card's slots. A text encoder's kind is
told by its Diffusers class or its name:

- **CLIP-L:** `CLIPTextModel`;
- **CLIP-G:** SDXL's and SD3's `text_encoder_2`;
- **T5-XXL:** `T5EncoderModel`, `*t5xxl*`;
- **LLM:** anything else, such as Qwen3 or Mistral.

`--model` takes a whole checkpoint (a file at the repo's top level, or in `checkpoints/`),
never a part from a Diffusers folder. A tiny autoencoder (`taesd`, `taef1`, `taeqi`) goes in
`--tae`, not `--vae`. Where a repo has both a named file (`ae.safetensors`) and a Diffusers part
of the same thing, the named file is suggested.

## Running a card

**Use on an engine** points an sd-server engine at the card (`model_card` in the profile's
`.local.yaml`):

- **An engine launched here** that is running restarts with the card's files. One that is
  stopped loads them when it is next started.
- **An engine started elsewhere:** the card only says what it runs, for its defaults,
  presets and capabilities.

Switching is refused while the engine is making an image.

While an engine runs a card:

- **Settings in use:** its config as it runs is its own, with the card's files, family,
  defaults and presets over it. The profile keeps only the card's id.
- **Editing files:** on the Engines screen, the Hugging Face dialog and the LoRA library's
  "add to preset", changes go to the card.
- **Applying changes:** new defaults and presets apply at once. New files apply when the
  engine restarts, and the card and Engines screen say when it needs one.

**Moving in.** An engine set up before the library has its files in its own launch block.
At startup it gets a card made from those files, with its family detected and its defaults,
presets and default LoRAs copied, and it is pointed at that card. Nothing it does changes.

## Settings a card can set

| Setting | Sent to sd-server as |
|---|---|
| `width`, `height` | `width`, `height` (snapped to the family's size multiple) |
| `steps`, `sampler`, `scheduler`, `flow_shift`, `sigmas` | `sample_params.sample_steps`, `sample_method`, `scheduler`, `flow_shift`, `custom_sigmas` |
| `cfg_scale` | `sample_params.guidance.txt_cfg`. At 1, CFG is off (distilled models) and the negative prompt has no effect |
| `guidance` | `sample_params.guidance.distilled_guidance` (FLUX.1 dev, FLUX.2 dev) |
| `clip_skip` | `clip_skip` (SD 1.x/2.x/SDXL) |
| `extra_sample` | merged into `sample_params` (e.g. NitroSD's `shifted_timestep`) |
| `negative_prompt` | `negative_prompt` |

Presets set `steps`, `cfg_scale`, `sampler`, `scheduler`, `flow_shift`, `sigmas`,
`guidance`, `clip_skip` and LoRAs, as before.
