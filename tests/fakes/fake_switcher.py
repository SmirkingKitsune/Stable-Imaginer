"""Stand-ins for an sd-server with an edit card: the switcher the engine service would be,
and an engine config with both cards (Qwen-Image-2.1, and Qwen-Image-Edit-2511 for edits)."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from imaginer.backends.base import BackendHealth
from imaginer.backends.sdcpp import Face
from imaginer.core.config import ImagePreset, SdCppBackendConfig, SdCppLaunch
from tests.conftest import sd_config


class FakeSwitcher:
    """Records which face asked for the engine, and can say it is switching."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Face, str | None]] = []
        self.switching: str | None = None

    def loading(self, engine: str) -> str | None:
        return self.switching

    def card_name(self, card_id: str | None) -> str | None:
        return f"Card {card_id}" if card_id else None

    async def ensure(
        self,
        engine: str,
        face: Face,
        config: SdCppBackendConfig,
        *,
        notify: Callable[[str | None], Awaitable[None]] | None,
        probe: Callable[[], Awaitable[BackendHealth]],
    ) -> None:
        self.calls.append((engine, face, config.model_card))
        if notify is not None:
            await notify(config.model_card)
            await notify(None)


def two_card_engine(url: str, **edit: Any) -> SdCppBackendConfig:
    """Qwen-Image-2.1 for text-to-image, Qwen-Image-Edit-2511 for edits."""
    launch = SdCppLaunch(diffusion_model=Path("main.gguf"))
    main = sd_config(
        url,
        family="qwen-image-21",
        model_card="main",
        edit_card="edit",
        launch=launch,
        presets={"draft": ImagePreset(steps=2)},
    )
    view = sd_config(
        url,
        family="qwen-image-edit-2511",
        model_card="edit",
        launch=SdCppLaunch(diffusion_model=Path("edit.gguf")),
        presets={"lightning": ImagePreset(steps=4, cfg_scale=1.0)},
        **edit,
    )
    return main.model_copy(update={"edit_view": view})
