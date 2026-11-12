"""Switching an sd-server between its model card and its edit card (EngineService.ensure):
a restart with the other card's files, then a wait until it has loaded them. Real
processes (a Python stand-in for sd-server); the load is faked by the probe."""

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from imaginer.backends.base import BackendHealth
from imaginer.backends.registry import BackendRegistry
from imaginer.backends.sdcpp import Face
from imaginer.core.clock import utcnow
from imaginer.core.config import RoutingConfig, SdCppBackendConfig, SdCppLaunch
from imaginer.domain.enums import BackendStatus
from imaginer.domain.errors import (
    EngineTimeoutError,
    EngineUnavailableError,
    InvalidRequestError,
)
from imaginer.engines import launch, service
from imaginer.engines.service import EngineService
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.live import dead_url

# A stand-in sd-server: says which file it loaded and stays up, unless it is told to die.
SCRIPT = (
    "import sys, time; print('loaded', sys.argv[1], flush=True); "
    "sys.exit(3) if 'dead' in sys.argv[1] else time.sleep(60)"
)


def _card(model: str, card: str, timeout: float = 5) -> SdCppBackendConfig:
    return sd_config(
        dead_url(),
        model_card=card,
        launch=SdCppLaunch(
            executable=Path(sys.executable), diffusion_model=Path(model), swap_timeout_s=timeout
        ),
    )


class Probe:
    """The engine's health as the switch sees it: loading until ``ready``."""

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.asked = asyncio.Event()  # the switch is waiting for the engine

    async def __call__(self) -> BackendHealth:
        self.asked.set()
        status = BackendStatus.OK if self.ready else BackendStatus.DEGRADED
        return BackendHealth(status=status, detail="", probed_at=utcnow())


@pytest.fixture
async def engines(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[EngineService]:
    def command(config: Any, exe: Path) -> list[str]:
        model = str(config.launch.diffusion_model)
        if "missing" in model:
            raise launch.LaunchError(f"--diffusion-model {model}: file not found")
        return [str(exe), "-c", SCRIPT, model]

    monkeypatch.setattr(launch, "command", command)
    monkeypatch.setattr(service, "SWITCH_POLL_S", 0.02)
    settings = make_settings(
        backends={"sd": _card("main.gguf", "main")}, routing=RoutingConfig(t2i="sd", edit="sd")
    )
    registry = BackendRegistry.from_settings(settings)
    engines = EngineService(settings, registry)
    engines.card_name = lambda card: f"Card {card}" if card else None
    yield engines
    await engines.supervisor.aclose()
    await registry.aclose()


async def _ensure(engines: EngineService, config: SdCppBackendConfig, probe: Probe) -> list[Any]:
    told: list[Any] = []

    async def notify(card: str | None) -> None:
        told.append(card)

    face: Face = "edit" if config.model_card == "edit" else "main"
    await engines.ensure("sd", face, config, notify=notify, probe=probe)
    return told


async def test_a_switch_restarts_the_engine_with_the_other_cards_files(
    engines: EngineService,
) -> None:
    await engines.start("sd")
    first = engines.supervisor.peek("sd")
    assert first is not None and first.run is not None
    started = first.run.argv
    told = await _ensure(engines, _card("edit.gguf", "edit"), Probe())
    assert told == ["Card edit", None]  # the job says what it waits for, then not
    assert engines.running_command("sd") == [sys.executable, "-c", SCRIPT, "edit.gguf"]
    assert engines.loading("sd") is None
    # Already running the edit card: nothing restarts.
    proc = engines.supervisor.peek("sd")
    assert proc is not None
    run = proc.run
    assert await _ensure(engines, _card("edit.gguf", "edit"), Probe()) == []
    assert proc.run is run
    # Text-to-image switches back.
    assert await _ensure(engines, _card("main.gguf", "main"), Probe()) == ["Card main", None]
    assert engines.running_command("sd") == started


async def test_a_stopped_engine_isnt_started_to_switch(engines: EngineService) -> None:
    with pytest.raises(EngineUnavailableError, match="is stopped"):
        await _ensure(engines, _card("edit.gguf", "edit"), Probe())
    assert not engines.running("sd")


async def test_missing_files_fail_before_anything_stops(engines: EngineService) -> None:
    await engines.start("sd")
    with pytest.raises(InvalidRequestError, match="can't switch cards"):
        await _ensure(engines, _card("missing.gguf", "edit"), Probe())
    assert engines.running_command("sd") == [sys.executable, "-c", SCRIPT, "main.gguf"]


async def test_a_card_that_never_loads_times_out(engines: EngineService) -> None:
    await engines.start("sd")
    with pytest.raises(EngineTimeoutError, match="did not load Card edit"):
        await _ensure(engines, _card("edit.gguf", "edit", timeout=0.2), Probe(ready=False))
    assert engines.loading("sd") is None


async def test_an_engine_that_dies_while_loading(engines: EngineService) -> None:
    await engines.start("sd")
    with pytest.raises(EngineUnavailableError, match="stopped while loading Card edit"):
        await _ensure(engines, _card("dead.gguf", "edit"), Probe(ready=False))
    assert engines.loading("sd") is None


async def test_a_cancelled_switch_leaves_the_engine_loading_its_card(
    engines: EngineService,
) -> None:
    await engines.start("sd")
    probe = Probe(ready=False)
    switch = asyncio.create_task(_ensure(engines, _card("edit.gguf", "edit"), probe))
    await probe.asked.wait()
    switch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await switch
    # The restart finished: the engine goes on loading the edit card, and says so.
    assert engines.running_command("sd") == [sys.executable, "-c", SCRIPT, "edit.gguf"]
    assert engines.loading("sd") == "Card edit"
    probe.ready = True
    for _ in range(200):
        if engines.loading("sd") is None:
            break
        await asyncio.sleep(0.01)
    assert engines.loading("sd") is None


async def test_stopping_the_engine_ends_the_wait(engines: EngineService) -> None:
    await engines.start("sd")
    probe = Probe(ready=False)
    switch = asyncio.create_task(_ensure(engines, _card("edit.gguf", "edit"), probe))
    await probe.asked.wait()
    await engines.stop("sd")  # the lock is free while it waits
    with pytest.raises(EngineUnavailableError, match="stopped while loading"):
        await switch
