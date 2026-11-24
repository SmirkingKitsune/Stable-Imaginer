"""An sd-server with an edit card: two faces on one server, one per card. Edits get the edit
face (its card's family, presets and limits), text-to-image the main one, and each makes
sure the engine runs its card first (the engine service switches it; faked here)."""

from pathlib import Path
from typing import Any

import pytest

from imaginer.backends.base import EditRequest, GenRequest, ImageInput
from imaginer.backends.registry import BackendRegistry
from imaginer.backends.sdcpp import Face, SdCppBackend, edit_face, edit_ref_limit, mask_face
from imaginer.core.config import ImagePreset, RoutingConfig, SdCppBackendConfig, SdCppLaunch
from imaginer.domain.enums import BackendStatus, Capability
from imaginer.domain.errors import EngineUnavailableError, InvalidRequestError
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.fakes.fake_switcher import FakeSwitcher, two_card_engine
from tests.fakes.live import dead_url
from tests.helpers import image_bytes

PHOTO = ImageInput(data=image_bytes(), mime="image/png")


async def test_each_face_runs_its_own_card(fake_sd: FakeSdCpp) -> None:
    switcher = FakeSwitcher()
    backend = SdCppBackend("sd", two_card_engine(fake_sd.url))
    edit = backend.edit_face
    assert edit is not None and edit.family == "qwen-image-edit-2511"
    backend.switcher = edit.switcher = switcher
    await backend.health()
    assert Capability.T2I in backend.capabilities()
    assert edit.capabilities() == {Capability.EDIT, Capability.MULTI_REF}  # the edit model's
    assert edit.max_ref_images == 3 and not edit.takes_mask
    await backend.generate(GenRequest(prompt="a fox", seed=1))
    await edit.edit(EditRequest(prompt="make it red", seed=1, images=(PHOTO,), preset="lightning"))
    assert [face for _, face, _ in switcher.calls] == ["main", "edit"]
    assert fake_sd.requests[-1]["sample_params"]["sample_steps"] == 4  # the edit card's preset
    assert edit.model_card == "edit" and backend.model_card == "main"
    with pytest.raises(InvalidRequestError, match="can't edit with a mask"):
        await edit.edit(EditRequest(prompt="x", seed=1, images=(PHOTO,), mask=PHOTO))
    await backend.aclose()


async def test_while_it_switches_the_engine_says_so_and_keeps_its_capabilities() -> None:
    switcher = FakeSwitcher()
    backend = SdCppBackend("sd", two_card_engine(dead_url()))
    backend.switcher = switcher
    health = await backend.health()
    assert health.status is BackendStatus.DOWN and backend.capabilities() == set()
    switcher.switching = "Qwen Image Edit 2511"
    health = await backend.health()
    assert (
        health.status is BackendStatus.DEGRADED and health.detail == "loading Qwen Image Edit 2511"
    )
    assert Capability.T2I in backend.capabilities()  # what it will do once loaded
    await backend.aclose()


async def test_the_registry_leases_the_face_a_task_needs(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    settings = make_settings(
        backends={"sd": two_card_engine(fake_sd.url)}, routing=RoutingConfig(t2i="sd", edit="sd")
    )
    registry = BackendRegistry.from_settings(settings)
    switcher = FakeSwitcher()
    registry.set_switcher(switcher)
    t2i = await registry.image_backend("t2i")
    edit = await registry.image_backend("edit", preset="lightning")
    assert isinstance(edit.backend, SdCppBackend) and edit.backend.face == "edit"
    assert isinstance(t2i.backend, SdCppBackend) and t2i.backend.face == "main"
    assert edit.backend.switcher is switcher  # attached to both faces
    # A preset belongs to its card: the edit card's isn't the text-to-image one's.
    with pytest.raises(InvalidRequestError, match="lightning"):
        await registry.image_backend("t2i", preset="lightning")
    registry.validate_image_preset("lightning")  # a message may name either card's
    registry.validate_image_preset("draft")
    assert registry.image_preset("edit", None, "lightning") is not None
    assert registry.image_preset("t2i", None, "lightning") is None
    # Either face's lease holds the engine.
    t2i.release()
    assert registry.leased("sd")
    edit.release()
    assert not registry.leased("sd")
    info = (await registry.describe())[0]
    assert info.edit is not None and info.edit.card == "edit"
    assert info.edit.card_name == "Card edit"
    # The edit card takes 3 images; edits with more run on Qwen-Image-2.1, which takes 10.
    assert (info.edit.max_ref_images, info.edit.card_max_ref_images) == (10, 3)
    assert (info.edit.refs_card, info.edit.refs_card_name) == ("main", "Card main")
    assert [p.name for p in info.edit.image_presets] == ["lightning"]
    assert [p.name for p in info.image_presets] == ["draft"]
    assert Capability.MULTI_REF in info.capabilities  # the engine edits with several images
    await registry.aclose()


async def test_a_masked_edit_runs_on_the_model_card_when_only_it_takes_a_mask(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    # Qwen-Image-Edit-2511 takes no mask; Qwen-Image-2.1 edits with one.
    settings = make_settings(
        backends={"sd": two_card_engine(fake_sd.url)}, routing=RoutingConfig(t2i="sd", edit="sd")
    )
    registry = BackendRegistry.from_settings(settings)
    switcher = FakeSwitcher()
    registry.set_switcher(switcher)
    lease = await registry.image_backend("edit", require={Capability.MULTI_REF}, mask=True)
    assert isinstance(lease.backend, SdCppBackend) and lease.backend.face == "main"
    await lease.backend.edit(EditRequest(prompt="a red sky", seed=1, images=(PHOTO,), mask=PHOTO))
    lease.release()
    assert [face for _, face, _ in switcher.calls] == ["main"]  # the engine runs the model card
    body = fake_sd.requests[-1]
    assert len(body["ref_images"]) == 2 and "mask" in body["prompt"]  # Qwen-Image-2.1's way
    # Its presets are the model card's; the edit card's are for edits without a mask.
    assert registry.image_preset("edit", None, "draft", mask=True) is not None
    assert registry.image_preset("edit", None, "lightning", mask=True) is None
    with pytest.raises(InvalidRequestError, match="lightning"):
        await registry.image_backend("edit", preset="lightning", mask=True)
    plain = await registry.image_backend("edit")
    assert isinstance(plain.backend, SdCppBackend) and plain.backend.face == "edit"
    plain.release()
    info = (await registry.describe())[0]
    assert info.edit is not None and info.edit.mask  # the engine can still mask
    assert (info.edit.mask_card, info.edit.mask_card_name) == ("main", "Card main")
    await registry.aclose()


def test_a_masked_edit_stays_on_the_edit_card_unless_only_the_model_card_takes_masks() -> None:
    url = dead_url()
    assert mask_face(two_card_engine(url)) == "main"
    # An edit card that takes a mask (here, by its config) keeps masked edits.
    masking = two_card_engine(url, mask_instruction="White is the area to change.")
    assert mask_face(masking) == "edit"
    # A model card that takes none either: the edit card refuses the mask, as it would alone.
    flux = sd_config(
        url,
        family="flux1-dev",
        model_card="main",
        edit_card="edit",
        launch=SdCppLaunch(diffusion_model=Path("main.gguf")),
    )
    kontext = sd_config(
        url,
        family="flux1-kontext",
        model_card="edit",
        launch=SdCppLaunch(diffusion_model=Path("edit.gguf")),
    )
    assert mask_face(flux.model_copy(update={"edit_view": kontext})) == "edit"
    assert mask_face(sd_config(url, family="qwen-image-21")) is None  # no edit card


def test_an_edit_with_more_images_than_the_edit_card_takes_runs_on_the_model_card() -> None:
    url = dead_url()
    # FLUX.1 Kontext edits one image; Qwen-Image-2.1, the model card, up to ten.
    kontext = two_card_engine(url, family="flux1-kontext")
    assert edit_face(kontext) == "edit"  # one image: Kontext's
    assert edit_face(kontext, images=2) == "main"
    assert edit_face(kontext, images=10) == "main"
    assert edit_face(kontext, images=11) == "edit"  # neither takes it: the edit card refuses
    assert edit_face(kontext, images=9, mask=True) == "main"
    assert edit_ref_limit(kontext) == 10
    # Qwen-Image-Edit-2511 takes three: the fourth goes to the model card.
    qwen_edit = two_card_engine(url)
    assert edit_face(qwen_edit, images=3) == "edit" and edit_face(qwen_edit, images=4) == "main"
    # A model card that doesn't edit leaves every edit to the edit card.
    flux = sd_config(
        url,
        family="flux1-dev",
        model_card="main",
        edit_card="edit",
        launch=SdCppLaunch(diffusion_model=Path("main.gguf")),
    )
    alone = flux.model_copy(update={"edit_view": kontext.edit_view})
    assert edit_face(alone, images=2) == "edit" and edit_ref_limit(alone) == 1
    # So does one whose limit is lower: the edit card takes more.
    narrow = kontext.model_copy(update={"max_ref_images": 1})
    assert edit_face(narrow, images=2) == "edit"
    assert edit_ref_limit(sd_config(url, family="qwen-image-21")) == 10  # no edit card


async def test_the_registry_routes_a_several_image_edit_to_the_model_card(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    settings = make_settings(
        backends={"sd": two_card_engine(fake_sd.url, family="flux1-kontext")},
        routing=RoutingConfig(t2i="sd", edit="sd"),
    )
    registry = BackendRegistry.from_settings(settings)
    switcher = FakeSwitcher()
    registry.set_switcher(switcher)
    one = await registry.image_backend("edit")
    assert isinstance(one.backend, SdCppBackend) and one.backend.face == "edit"
    one.release()
    two = await registry.image_backend("edit", require={Capability.MULTI_REF}, images=2)
    assert isinstance(two.backend, SdCppBackend) and two.backend.face == "main"
    await two.backend.edit(EditRequest(prompt="combine", seed=1, images=(PHOTO, PHOTO)))
    two.release()
    assert [face for _, face, _ in switcher.calls] == ["main"]  # the engine runs the model card
    assert len(fake_sd.requests[-1]["ref_images"]) == 2
    # Such an edit takes the model card's presets, not the edit card's.
    assert registry.image_preset("edit", None, "draft", images=2) is not None
    assert registry.image_preset("edit", None, "lightning", images=2) is None
    assert registry.image_preset("edit", None, "lightning") is not None
    info = (await registry.describe())[0]
    assert info.edit is not None and Capability.MULTI_REF in info.edit.capabilities
    assert (info.edit.max_ref_images, info.edit.card_max_ref_images) == (10, 1)
    assert (info.edit.refs_card, info.edit.refs_card_name) == ("main", "Card main")
    await registry.aclose()


async def test_work_waits_for_a_switching_engine_but_not_a_stopped_one(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(
        backends={"sd": two_card_engine(dead_url())}, routing=RoutingConfig(t2i="sd", edit="sd")
    )
    registry = BackendRegistry.from_settings(settings)
    switcher = FakeSwitcher()
    registry.set_switcher(switcher)
    with pytest.raises(EngineUnavailableError):
        await registry.image_backend("edit")
    switcher.switching = "Qwen Image Edit 2511"
    lease = await registry.image_backend("edit", require={Capability.MULTI_REF})
    assert lease.backend.capabilities() >= {Capability.EDIT, Capability.MULTI_REF}
    lease.release()
    await registry.aclose()


async def test_an_engine_with_only_an_edit_model_offers_no_text_to_image(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    settings = make_settings(
        backends={"sd": sd_config(fake_sd.url, family="qwen-image-edit-2511")},
        routing=RoutingConfig(t2i="sd", edit="sd"),
    )
    registry = BackendRegistry.from_settings(settings)
    assert registry.offers("edit") and not registry.offers("t2i")
    await registry.aclose()


async def test_changing_the_edit_cards_settings_takes_them_as_they_are(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    config = two_card_engine(fake_sd.url)
    settings = make_settings(backends={"sd": config}, routing=RoutingConfig(t2i="sd", edit="sd"))
    registry = BackendRegistry.from_settings(settings)
    adapter = registry._entries["sd"].backend
    assert config.edit_view is not None
    presets = {"lightning": ImagePreset(steps=6)}
    view = config.edit_view.model_copy(update={"presets": presets})
    assert not registry.reconfigure("sd", config.model_copy(update={"edit_view": view}))
    assert registry._entries["sd"].backend is adapter  # presets are read per request
    other = view.model_copy(update={"family": "qwen-image-edit-2509"})
    assert registry.reconfigure("sd", config.model_copy(update={"edit_view": other}))
    await registry.aclose()


async def test_a_card_whose_files_changed_is_switched_to_with_its_new_files(
    make_settings: SettingsFactory, fake_sd: FakeSdCpp
) -> None:
    """A model card's files fixed while the engine runs: the adapter is kept, and a card
    switch starts the engine with the new files, never the old ones."""
    config = two_card_engine(fake_sd.url)
    settings = make_settings(backends={"sd": config}, routing=RoutingConfig(t2i="sd", edit="sd"))
    registry = BackendRegistry.from_settings(settings)
    adapter = registry._entries["sd"].backend
    assert isinstance(adapter, SdCppBackend) and adapter.edit_face is not None
    assert config.launch is not None and config.edit_view is not None
    fixed = config.model_copy(
        update={
            "launch": config.launch.model_copy(update={"llm": Path("right-encoder.gguf")}),
            "edit_view": config.edit_view.model_copy(
                update={
                    "launch": SdCppLaunch(
                        diffusion_model=Path("edit.gguf"), llm=Path("edit-encoder.gguf")
                    )
                }
            ),
        }
    )
    assert not registry.reconfigure("sd", fixed)
    assert registry._entries["sd"].backend is adapter
    assert adapter.config.launch is not None and adapter.edit_face.config.launch is not None
    assert adapter.config.launch.llm == Path("right-encoder.gguf")
    assert adapter.edit_face.config.launch.llm == Path("edit-encoder.gguf")
    switcher = SwitchedWith()
    adapter.switcher = switcher
    await adapter.generate(GenRequest(prompt="a fox", seed=1))
    assert switcher.llms == [Path("right-encoder.gguf")]
    await registry.aclose()


class SwitchedWith(FakeSwitcher):
    """Records the text encoder each switch would start the engine with."""

    def __init__(self) -> None:
        super().__init__()
        self.llms: list[Path | None] = []

    async def ensure(self, engine: str, face: Face, config: SdCppBackendConfig, **kw: Any) -> None:
        self.llms.append(config.launch.llm if config.launch is not None else None)
        await super().ensure(engine, face, config, **kw)
