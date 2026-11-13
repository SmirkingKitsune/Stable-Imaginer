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

Control, inpainting and decoder recipes (ControlNets, PiD…) and the community-LoRA edit
recipes (Krea 2 edit, Anima edit) are hidden from the library. Video recipes are not included.

What a family can do follows its recipe and our notes:

- edit and multi-reference need its `ref_images` feature (Qwen-Image-2.1, FLUX.2, FLUX.3,
  Fibo, P-H3, and the edit models below);
- several images in one edit need a family that handles more than one (`max_refs`);
- transparent output needs a prompt template (Qwen-Image-2.1, Ming-Image);
- an edit with a mask needs a mask instruction (Qwen-Image-2.1).

An engine running a family that lacks one of these doesn't offer it.

## Edit models

sd.cpp's edit models make an image from the images they are given (stable-diffusion.cpp
`docs/edit.md`). Their cards are made like any other. Some only edit (`edit_only`): an engine
running one offers no text-to-image, and the chat model gets no `generate_image` tool.

| Family | Makes images from text | Images per edit | Notes |
|---|---|---|---|
| FLUX.1 Kontext dev | yes | 1 | CLIP-L, T5 and the FLUX VAE, as FLUX.1 dev; distilled guidance 2.5 |
| Qwen Image Edit | no | 1 | Qwen 2.5 VL 7B text encoder, the Qwen Image VAE |
| Qwen Image Edit 2509 / 2511 | no | 3 | the same files; "Picture 1, Picture 2" in prompts. 2511 gets `qwen_image_zero_cond_t=true` |
| LongCat Image Edit | no | 1 | Qwen 2.5 VL 7B, the FLUX VAE |
| Boogu Image Edit | no | any | Qwen3-VL 8B, the FLUX VAE |
| Mage-Flow Edit | no | any | Qwen3-VL 4B, its own VAE; "Image 1, Image 2" |
| Fibo 2.0 Edit Turbo | no | any | Qwen3-VL 4B, the FLUX.2 VAE; 4 steps at CFG 1 |
| Fibo 2.0 Dev, LLaDA-Image | yes | any / 1 | LLaDA edits with its `connectors-edit` file, at sizes in multiples of 32 |

How it works:

- **Detection** tells an edit model from the text-to-image one it shares an architecture or a
  Hub repo with: Kontext has FLUX.1 dev's tensors, and Comfy-Org's repos hold both. Its file
  name, repo or Diffusers pipeline (`FluxKontextPipeline`, `QwenImageEditPlusPipeline`…) decides.
- **The vision projector:** an edit model reads the images with its text encoder's vision
  weights. A safetensors encoder (`qwen_2.5_vl_7b.safetensors`) has them; a GGUF one needs its
  mmproj in `llm_vision`. Without it an edit-only card isn't ready, and a card that also makes
  images from text can't edit (its card says so).
- **Files are shared** with the cards of the same group only where both families' recipes name
  the slot alike: a Qwen Image Edit card is offered Qwen-Image's 2.5-VL encoder, never
  Qwen-Image-2.1's 3-VL one.
- **Image guidance** (`img_cfg_scale`, sd.cpp's `--img-cfg-scale`) sets how closely an edit keeps
  to its images, for the families with image CFG (Kontext, the Qwen edits, LongCat, Boogu,
  Mage-Flow). Unset, it is 1.
- **Reference image args** (`ref_image_args`) pass sd-server's per-request settings for how the
  images are fed in, e.g. `resize_before_vae=false` or `vlm_max_size=512` (sd.cpp `docs/edit.md`).
- **LoRAs** made for an edit model (their base model names Kontext, Qwen-Image-Edit…) are
  grouped with it, not with the text-to-image model.

### An edit card: one engine, two models

An sd-server engine runs one card. To edit with an edit model next to the model you make
images with, give the engine an **edit card** ("Use for edits on <engine>" on the card, or
"Edits with" on the Engines screen; `edit_card` in the profile):

- an edit restarts the engine with the edit card's files and waits until it answers
  (`launch.swap_timeout_s`, 300 s by default). sd-server answers before it has read the
  weights, and reads them as the first image starts, so the job shows "Loading <card>" until
  that image's first step (on a 5090, about a minute for Qwen-Image-Edit-2511 Q8_0 from disk;
  seconds once it is cached). A family without previews reports no steps: its label ends when
  the engine answers;
- the next text-to-image request switches it back. Consecutive edits (or images) don't switch;
- each card keeps its own defaults and presets. The prompt bar offers the edit card's presets
  too, marked "edits": a preset applies to the task its card serves;
- an edit with a mask runs on the model card when the edit card takes no mask and the model
  card does: Qwen-Image-2.1 next to Kontext or Qwen-Image-Edit keeps Mask available. That edit
  uses the model card's presets, and the Mask button's tooltip names the card;
- the chat model is told how to prompt both;
- only an engine launched here switches (it needs `launch`), one image at a time
  (`concurrency: 1`). A stopped engine stays stopped: start it first. Which card is loaded is
  not saved: the engine always starts with its model card.

Each switch loads a model, which takes seconds to minutes. With the VRAM for both, a second
engine routed for edits (`routing.edit`) avoids the switching.

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
| `guidance` | `sample_params.guidance.distilled_guidance` (FLUX.1 dev, Kontext, FLUX.2 dev) |
| `img_cfg_scale` | `sample_params.guidance.img_cfg`, edits only, for families with image CFG |
| `ref_image_args` | `ref_image_args`, edits only (e.g. `resize_before_vae=false`) |
| `clip_skip` | `clip_skip` (SD 1.x/2.x/SDXL) |
| `extra_sample` | merged into `sample_params` (e.g. NitroSD's `shifted_timestep`) |
| `negative_prompt` | `negative_prompt` |

Presets set `steps`, `cfg_scale`, `sampler`, `scheduler`, `flow_shift`, `sigmas`,
`guidance`, `clip_skip`, `img_cfg_scale` and LoRAs.
