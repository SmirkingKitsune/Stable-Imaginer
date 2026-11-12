"""An sd-server with an edit card: two faces on one server, one per card. Edits get the edit
face (its card's family, presets and limits), text-to-image the main one, and each makes
sure the engine runs its card first (the engine service switches it; faked here)."""

import pytest

from imaginer.backends.base import EditRequest, GenRequest, ImageInput
from imaginer.backends.registry import BackendRegistry
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.config import ImagePreset, RoutingConfig
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
    assert info.edit.card_name == "Card edit" and info.edit.max_ref_images == 3
    assert [p.name for p in info.edit.image_presets] == ["lightning"]
    assert [p.name for p in info.image_presets] == ["draft"]
    assert Capability.MULTI_REF in info.capabilities  # the engine edits with several images
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
