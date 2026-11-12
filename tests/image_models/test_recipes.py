"""The model families: sd-server's recipes (the bundled snapshot) with our notes on them."""

from imaginer.domain.enums import Capability
from imaginer.image_models import recipes
from imaginer.image_models.library import seed_defaults

EDIT_FAMILIES = (
    "flux1-kontext",
    "qwen-image-edit",
    "qwen-image-edit-2509",
    "qwen-image-edit-2511",
    "longcat-image-edit",
    "boogu-image-edit",
    "mage-flow-edit",
    "fibo2-edit-turbo",
)


def test_every_image_recipe_is_a_family_and_control_ones_are_hidden() -> None:
    catalog = recipes.catalog()
    shown = {f.id for f in catalog.families()}
    every = {f.id for f in catalog.families(hidden=True)}
    assert {
        "sd1-checkpoint",
        "sdxl-checkpoint",
        "flux1-dev",
        "qwen-image-21",
        "z-image-turbo",
        *EDIT_FAMILIES,
    } <= shown
    hidden = every - shown
    assert {"pid", "sdxl-controlnet", "krea2-raw-edit", "qwen-image-layered"} <= hidden
    assert not any(f.id.startswith(("wan2", "ltx")) for f in catalog.families(hidden=True))
    # Every family shown is grouped and placed for LoRAs.
    assert all(f.group and f.lora_group for f in catalog.families())


def test_a_family_says_what_it_needs() -> None:
    flux = recipes.family("flux1-dev")
    assert flux is not None
    assert flux.slot_ids == ("diffusion_model", "clip_l", "t5xxl", "vae", "tae")
    assert flux.main_slot == "diffusion_model"
    assert flux.missing(["diffusion_model"]) == ["clip_l", "t5xxl", "vae or tae"]
    assert flux.missing(["diffusion_model", "clip_l", "t5xxl", "tae"]) == []
    assert flux.defaults["guidance"] == 3.5 and flux.defaults["sampler"] == "euler"
    sdxl = recipes.family("sdxl-checkpoint")
    assert sdxl is not None and sdxl.main_slot == "model" and sdxl.missing([]) == ["model"]
    assert sdxl.size_multiple == 64


def test_capabilities_follow_the_recipe_and_our_notes() -> None:
    qwen = recipes.family("qwen-image-21")
    flux = recipes.family("flux1-dev")
    klein = recipes.family("flux2-klein-4b")
    assert qwen is not None and flux is not None and klein is not None
    assert qwen.capabilities() == {
        Capability.T2I, Capability.LORA, Capability.EDIT, Capability.MULTI_REF, Capability.RGBA
    }  # fmt: skip
    assert qwen.mask is not None
    assert flux.capabilities() == {Capability.T2I, Capability.LORA}
    assert Capability.EDIT in klein.capabilities() and Capability.RGBA not in klein.capabilities()


def test_new_cards_start_from_the_family_defaults() -> None:
    qwen = seed_defaults("qwen-image-21")
    assert (qwen.width, qwen.height, qwen.cfg_scale) == (2048, 2048, 1)  # native 2K
    nitro = seed_defaults("nitrosd-realism")
    assert nitro.extra_sample == {"shifted_timestep": 250} and nitro.scheduler == "sgm_uniform"
    turbo = seed_defaults("qwen-image-21-turbo")
    assert turbo.sigmas is not None and turbo.sigmas[-1] == 0.0 and turbo.steps == 8
    l2p = seed_defaults("z-image-l2p")  # its recipe names no size: our notes do
    assert (l2p.width, l2p.cfg_scale, l2p.steps) == (1024, 2.0, 30)
    assert seed_defaults(None).width == 1024


def test_prompt_hints_and_model_args() -> None:
    sdxl, ideogram, chroma = (recipes.family(i) for i in ("sdxl-checkpoint", "ideogram4", "chroma"))
    assert sdxl is not None and "comma-separated tags" in (sdxl.prompt_hint() or "")
    assert ideogram is not None and "JSON" in (ideogram.prompt_hint() or "")
    assert recipes.family("flux1-dev").prompt_hint() is None  # type: ignore[union-attr]
    assert chroma is not None and chroma.model_args == {"chroma_use_dit_mask": "false"}
    pixart = recipes.family("pixart-alpha-512")
    assert pixart is not None and pixart.model_args["pixart_pos_embed_base_size"] == "32"
    assert recipes.model_args_text({"a": "1", "b": "true"}) == "a=1,b=true"


def test_an_engines_recipes_replace_the_snapshots() -> None:
    catalog = recipes.Catalog(
        [{"id": "x-image", "name": "X", "slots": [{"id": "diffusion_model", "required": True}]}],
        {"families": {"x-image": {"group": "X"}}},
    )
    taken = catalog.take_engine_recipes(
        [
            {
                "schema_version": 1,
                "id": "x-image",
                "name": "X Image",
                "source": "builtin",
                "effective_features": {"ref_images": True},
                "slots": [
                    {"id": "diffusion_model", "required": True},
                    {"id": "llm", "required": True},
                ],
                "features": ["ref_images"],
            },
            {"id": "some-video", "name": "V", "slots": [], "video_constraints": {}},
            "not a recipe",
        ]
    )
    assert taken == 1
    family = catalog.get("x-image")
    assert family is not None and family.origin == "engine" and family.group == "X"
    assert family.slot_ids == ("diffusion_model", "llm") and "ref_images" in family.features
    assert catalog.get("some-video") is None


def test_edit_families_say_what_they_take() -> None:
    kontext, qwen, edit_2511, longcat = (
        recipes.family(i)
        for i in ("flux1-kontext", "qwen-image-edit", "qwen-image-edit-2511", "longcat-image-edit")
    )
    assert kontext is not None and qwen is not None and edit_2511 is not None
    assert longcat is not None
    # Kontext makes images from text too, and edits one image at a time.
    assert kontext.capabilities() == {Capability.T2I, Capability.LORA, Capability.EDIT}
    assert kontext.group == "FLUX.1" and kontext.lora_group == "flux1-kontext"
    # The Qwen edit models only edit; 2509 and 2511 take several images.
    assert edit_2511.capabilities() == {Capability.EDIT, Capability.MULTI_REF, Capability.LORA}
    assert qwen.capabilities() == {Capability.EDIT, Capability.LORA}
    assert edit_2511.model_args == {"qwen_image_zero_cond_t": "true"}
    assert "Picture 1" in (edit_2511.prompt_hint() or "")
    # LongCat edit's doc passes a GGUF encoder's mmproj, which its recipe has no slot for.
    assert longcat.slot("llm_vision") is not None and not longcat.slot("llm_vision").required  # type: ignore[union-attr]
    llada = recipes.family("llada-image")
    assert llada is not None and llada.size_multiple == 32 and llada.max_refs == 1


def test_an_edit_needs_the_text_encoders_vision() -> None:
    edit = recipes.family("qwen-image-edit-2511")
    assert edit is not None
    gguf = {"diffusion_model": "e.gguf", "llm": "Qwen2.5-VL-7B-Instruct.Q4_K_M.gguf"}
    assert edit.needs_vision(gguf)
    assert edit.capabilities(gguf) == {Capability.LORA}
    assert not edit.needs_vision({**gguf, "llm_vision": "mmproj-f16.gguf"})
    # A safetensors encoder carries its vision weights.
    assert not edit.needs_vision({"llm": "qwen_2.5_vl_7b.safetensors"})
    # Families that don't edit never need it.
    flux = recipes.family("flux1-dev")
    assert flux is not None and not flux.needs_vision({"llm": "x.gguf"})


def test_loras_made_for_an_edit_model_fit_it_not_the_text_to_image_one() -> None:
    from imaginer.loras.library import lora_group

    assert lora_group("black-forest-labs/FLUX.1-Kontext-dev", None) == "flux1-kontext"
    assert lora_group("black-forest-labs/FLUX.1-dev", None) == "flux1"
    assert lora_group("Qwen/Qwen-Image-Edit-2509", None) == "qwen-image-edit"
    assert lora_group("Qwen/Qwen-Image-2.1", None) == "qwen-image-21"
    assert lora_group(None, "qwen_image_edit") == "qwen-image-edit"
    # Every edit family's LoRA group is one a LoRA's base model can name.
    groups = {f.lora_group for f in recipes.catalog().families()}
    assert {"flux1-kontext", "qwen-image-edit", "mage-flow-edit", "fibo2-edit"} <= groups
