"""An sd-server failure comes with what the engine's process showed (its report): a crash
becomes ``engine_crashed`` and says why, rather than "unreachable". The engine service that
makes the report is faked here (tests/engines/test_engine_diagnostics.py tests the real one)."""

from datetime import timedelta

import pytest

from imaginer.backends.base import GenRequest
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.backends.registry import FACTORIES, BackendRegistry
from imaginer.backends.sdcpp import SdCppBackend
from imaginer.core.config import (
    BackendConfig,
    OrchestrationConfig,
    RoutingConfig,
    SdCppBackendConfig,
)
from imaginer.domain.errors import (
    EngineCrashedError,
    EngineError,
    EngineUnavailableError,
)
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.fake_reporter import ASSERT, FakeReporter
from tests.fakes.fake_sdcpp import FakeSdCpp
from tests.fakes.live import dead_url, refusing_client


def refusing_sd(name: str, config: BackendConfig) -> SdCppBackend:
    """An sd-server adapter whose engine refuses connections at once, as a crashed one does
    (a real refusal takes seconds on Windows)."""
    assert isinstance(config, SdCppBackendConfig)
    transport = HttpTransport(
        name=name,
        base_url=config.base_url,
        timeouts=config.timeouts,
        retry=config.retry,
        client=refusing_client(),
    )
    return SdCppBackend(name, config, transport=transport)


async def test_a_crash_while_making_the_image_says_why() -> None:
    backend = refusing_sd("sd", sd_config(dead_url(), family="anima"))
    backend.reporter = reporter = FakeReporter()
    with pytest.raises(EngineCrashedError) as caught:
        await backend.generate(GenRequest(prompt="a fox", seed=1))
    error = caught.value
    assert error.code == "engine_crashed"
    assert error.detail == f"sd crashed (exit code 3, aborted) while it made the image: {ASSERT}"
    assert error.extra["engine_report"]["hints"] == ["A tensor shape mismatch"]
    assert reporter.asked == [("sd", "anima", True)]  # it stopped answering: wait for its exit
    await backend.aclose()


async def test_a_job_the_running_engine_failed_keeps_its_error_and_gets_the_report(
    fake_sd: FakeSdCpp,
) -> None:
    fake_sd.fail_next = "failed to decode the latent"
    backend = SdCppBackend("sd", sd_config(fake_sd.url))
    backend.reporter = reporter = FakeReporter(state="running")
    await backend.health()
    with pytest.raises(EngineError) as caught:
        await backend.generate(GenRequest(prompt="a fox", seed=1))
    assert type(caught.value) is EngineError
    assert "failed to decode the latent" in caught.value.detail
    assert caught.value.extra["engine_report"]["state"] == "running"
    assert reporter.asked == [("sd", None, False)]  # it answered: no exit to wait for
    await backend.aclose()


async def test_an_engine_not_started_here_fails_as_before() -> None:
    backend = refusing_sd("sd", sd_config(dead_url()))
    with pytest.raises(EngineUnavailableError) as caught:
        await backend.generate(GenRequest(prompt="a fox", seed=1))
    assert "engine_report" not in caught.value.extra
    await backend.aclose()


async def test_the_next_request_to_a_crashed_engine_says_it_crashed_earlier(
    make_settings: SettingsFactory,
) -> None:
    settings = make_settings(
        backends={"sd": sd_config(dead_url())},
        routing=RoutingConfig(t2i="sd"),
        orchestration=OrchestrationConfig(probe_ttl_s=0),
    )
    registry = BackendRegistry.from_settings(settings, {**FACTORIES, "sdcpp": refusing_sd})
    registry.set_reporter(FakeReporter(ago=timedelta(minutes=2)))
    with pytest.raises(EngineCrashedError, match=r"sd isn't running: it crashed earlier \(exit"):
        await registry.image_backend("t2i")
    registry.set_reporter(FakeReporter(state="stopped"))
    with pytest.raises(EngineUnavailableError, match="sd is stopped: start it on the Engines"):
        await registry.image_backend("t2i")
    await registry.aclose()
