"""The Engines screen's backend: launch commands, model files, processes, installs, API."""

import asyncio
import struct
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI

from imaginer.core.config import SdCppLaunch, load_settings, local_overrides_path
from imaginer.engines import installer, launch, service
from imaginer.engines.supervisor import Supervisor
from imaginer.main import create_app
from tests.conftest import SettingsFactory, sd_config


def _gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + b"\0" * 60)
    return path


def _safetensors(path: Path) -> Path:
    header = b'{"w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    return path


# --- launch commands and model files ------------------------------------------------


def test_model_files_are_sniffed(tmp_path: Path) -> None:
    gguf = launch.check_file("model", "-m", "model", _gguf(tmp_path / "m.gguf"))
    assert (gguf.format, gguf.ok, gguf.size_bytes) == ("gguf", True, 64)
    tensors = launch.check_file("vae", "--vae", "VAE", _safetensors(tmp_path / "v.safetensors"))
    assert (tensors.format, tensors.ok) == ("safetensors", True)
    (tmp_path / "notes.txt").write_text("hello, not a model")
    junk = launch.check_file("vae", "--vae", "VAE", tmp_path / "notes.txt")
    assert (junk.format, junk.problem) == ("unknown", "not a GGUF or safetensors file")
    missing = launch.check_file("vae", "--vae", "VAE", tmp_path / "nope.gguf")
    assert (missing.exists, missing.problem) == (False, "file not found")
    assert launch.check_file("vae", "--vae", "VAE", None).ok is False


def test_sd_server_command(tmp_path: Path) -> None:
    config = sd_config(
        "http://127.0.0.1:1234",
        launch=SdCppLaunch(
            diffusion_model=_gguf(tmp_path / "diff.gguf"),
            vae=_safetensors(tmp_path / "vae.safetensors"),
            llm_vision=_gguf(tmp_path / "mmproj.gguf"),
            extra_args=["--threads", "8"],
        ),
    )
    argv = launch.command(config, Path("sd-server.exe"))
    assert argv == [
        "sd-server.exe",
        "--diffusion-model", str(tmp_path / "diff.gguf"),
        "--vae", str(tmp_path / "vae.safetensors"),
        "--llm_vision", str(tmp_path / "mmproj.gguf"),
        "--listen-ip", "127.0.0.1",
        "--listen-port", "1234",
        "--threads", "8",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("files", "fragment"),
    [
        ({}, "--diffusion-model (diffusion model) is not set"),
        ({"diffusion_model": "missing.gguf"}, "file not found"),
    ],
)
def test_launch_needs_usable_files(files: dict[str, str], fragment: str) -> None:
    config = sd_config("http://127.0.0.1:1234", launch=SdCppLaunch.model_validate(files))
    with pytest.raises(launch.LaunchError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        launch.command(config, Path("sd-server"))
    with pytest.raises(launch.LaunchError, match="no launch settings"):
        launch.command(sd_config("http://127.0.0.1:1234"), Path("sd-server"))


# --- processes -------------------------------------------------------------------------


async def _until(predicate: Any) -> None:
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.02)


async def test_a_failed_start_keeps_its_log_and_exit_code(tmp_path: Path) -> None:
    supervisor = Supervisor(tmp_path / "logs")
    script = (
        "print('loading vae', flush=True); print('error: wrong shape', flush=True); "
        "raise SystemExit(1)"
    )
    proc = await supervisor.start("sd", [sys.executable, "-c", script])
    await _until(lambda: proc.state == "exited")
    assert proc.run is not None and proc.run.exit_code == 1
    assert proc.run.last_error == "error: wrong shape"
    texts = [line.text for line in proc.run.lines]
    assert texts[1:] == ["loading vae", "error: wrong shape"]
    assert "error: wrong shape" in proc.run.log_path.read_text(encoding="utf-8")


async def test_stop_ends_a_running_engine(tmp_path: Path) -> None:
    supervisor = Supervisor(tmp_path / "logs")
    proc = await supervisor.start(
        "sd", [sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(60)"]
    )
    await _until(lambda: any(line.text == "ready" for line in proc.run.lines))  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="already running"):
        await supervisor.start("sd", [sys.executable, "-c", "pass"])
    await supervisor.stop("sd")
    assert proc.state == "stopped" and proc.run is not None and proc.run.stopped_by_us
    await supervisor.aclose()


SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture
def slow_spawn(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[str]]:
    """Process creation that takes a while (as it can on Windows): a window to cancel in."""
    real = subprocess.Popen
    spawned: list[subprocess.Popen[str]] = []

    def popen(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        time.sleep(0.3)
        spawned.append(real(*args, **kwargs))
        return spawned[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    return spawned


async def test_a_start_cancelled_while_spawning_leaves_nothing_running(
    tmp_path: Path, slow_spawn: list[subprocess.Popen[str]]
) -> None:
    supervisor = Supervisor(tmp_path / "logs")
    starting = asyncio.create_task(supervisor.start("sd", SLEEPER))
    await asyncio.sleep(0.05)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    proc = supervisor.get("sd")
    await _until(lambda: proc.state == "stopped")
    [spawned] = slow_spawn
    assert spawned.poll() is not None


async def test_shutdown_stops_an_engine_that_is_still_being_spawned(
    tmp_path: Path, slow_spawn: list[subprocess.Popen[str]]
) -> None:
    supervisor = Supervisor(tmp_path / "logs")
    starting = asyncio.create_task(supervisor.start("sd", SLEEPER))
    await asyncio.sleep(0.05)
    await supervisor.aclose()
    await starting
    assert supervisor.get("sd").state == "stopped"
    [spawned] = slow_spawn
    assert spawned.poll() is not None


# --- installs ----------------------------------------------------------------------------


class _FakeProvisioner:
    def __init__(self, config: Any, log_sink: Any, progress_sink: Any, cancel_check: Any) -> None:
        self.config, self.log_sink, self.progress_sink = config, log_sink, progress_sink
        self.cancel_check = cancel_check
        self.target_mismatch = ""
        self.build_log_path = None

    def ensure_runtime(self) -> Path:
        for step in (1, 2):
            self.progress_sink(
                SimpleNamespace(step=step, total=2, label=f"step {step}", fraction=0.5)
            )
            self.log_sink(f"working on {step}", False)
        if self.config.version == "slow":
            while not self.cancel_check():
                time.sleep(0.01)
            time.sleep(0.2)  # stopping a build, releasing the lock, undoing the swap
            (Path(self.config.provision_dir) / "cleaned-up").touch()
            raise RuntimeError("cancelled")
        return Path("bin/sd-server")


@pytest.fixture
def fake_provisioner(monkeypatch: pytest.MonkeyPatch) -> None:
    real = installer.engine_module

    def fake(engine_type: str) -> tuple[Any, type[Any]]:
        module, _ = real(engine_type)
        return module, _FakeProvisioner

    monkeypatch.setattr(installer, "engine_module", fake)


async def test_install_reports_progress_and_can_be_cancelled(
    tmp_path: Path, fake_provisioner: None
) -> None:
    runner = installer.Installer()
    seen: list[tuple[str, int]] = []
    runner.subscribe("sdcpp", lambda job: seen.append((job.state, job.step)))
    job = runner.start("sdcpp", tmp_path, installer.InstallRequest(accelerator="cpu"))
    await _until(lambda: bool(seen) and seen[-1][0] != "running")
    assert (job.state, job.executable, job.step, job.total) == (
        "succeeded",
        str(Path("bin/sd-server")),
        2,
        2,
    )
    assert list(job.output) == ["working on 1", "working on 2"]
    assert seen[-1] == ("succeeded", 2)

    slow = runner.start(
        "sdcpp", tmp_path, installer.InstallRequest(accelerator="cpu", version="slow")
    )
    with pytest.raises(RuntimeError, match="already running"):
        runner.start("sdcpp", tmp_path, installer.InstallRequest(accelerator="cpu"))
    runner.cancel("sdcpp")
    await _until(lambda: slow.state != "running")
    assert slow.state == "cancelled"


async def test_shutdown_waits_for_a_cancelled_install_to_clean_up(
    tmp_path: Path, fake_provisioner: None
) -> None:
    runner = installer.Installer()
    job = runner.start(
        "sdcpp", tmp_path, installer.InstallRequest(accelerator="cpu", version="slow")
    )
    await asyncio.sleep(0.05)
    await runner.aclose()
    assert job.state == "cancelled"
    assert (tmp_path / "cleaned-up").exists()


# --- API ---------------------------------------------------------------------------------

_FAST = {"retry": {"attempts": 1}, "timeouts": {"connect": 0.5}}  # its engines are never up


@pytest.fixture
def profile(tmp_path: Path) -> Path:
    path = tmp_path / "config" / "box.yaml"
    path.parent.mkdir()
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(tmp_path / "data"),
                "storage": {"sweep_interval_s": 0},
                "logging": {"level": "WARNING"},
                "backends": {
                    "sd": {
                        "type": "sdcpp",
                        "base_url": "http://127.0.0.1:1",
                        "declared_capabilities": ["t2i"],
                        "launch": {"executable": sys.executable},
                        **_FAST,
                    },
                    "remote": {
                        "type": "llamacpp",
                        "base_url": "http://127.0.0.2:9/inference/llama",
                        **_FAST,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
async def engines_client(profile: Path) -> AsyncIterator[httpx.AsyncClient]:
    app: FastAPI = create_app(load_settings(str(profile)))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_engines_are_listed_with_their_files(engines_client: httpx.AsyncClient) -> None:
    engines = {e["name"]: e for e in (await engines_client.get("/v1/engines")).json()}
    sd, remote = engines["sd"], engines["remote"]
    assert (sd["launchable"], sd["loopback"], sd["host"], sd["port"]) == (
        True,
        True,
        "127.0.0.1",
        1,
    )
    assert [f["flag"] for f in sd["files"]] == [
        "--diffusion-model",
        "--vae",
        "--llm",
        "--llm_vision",
    ]
    assert all(f["path"] is None for f in sd["files"])
    assert sd["runtime"]["installed"] is False
    assert (remote["launchable"], remote["loopback"], remote["files"][0]["flag"]) == (
        False,
        True,  # 127.0.0.2: remote only in that it has no launch settings
        "-m",
    )
    assert (await engines_client.get("/v1/engines/nope")).status_code == 404


async def test_changes_are_applied_and_saved_beside_the_profile(
    engines_client: httpx.AsyncClient, profile: Path, tmp_path: Path
) -> None:
    model = _gguf(tmp_path / "qwen.gguf")
    resp = await engines_client.patch(
        "/v1/engines/sd",
        json={
            "port": 1235,
            "declared_capabilities": ["t2i", "edit"],
            "launch": {"files": {"diffusion_model": str(model), "vae": str(tmp_path / "no.st")}},
        },
    )
    assert resp.status_code == 200, resp.text
    sd = resp.json()
    assert sd["base_url"] == "http://127.0.0.1:1235"
    assert sd["declared_capabilities"] == ["edit", "t2i"]
    files = {f["key"]: f for f in sd["files"]}
    assert files["diffusion_model"]["format"] == "gguf"
    assert files["vae"]["problem"] == "file not found"

    saved = yaml.safe_load(local_overrides_path(profile).read_text(encoding="utf-8"))
    assert saved["backends"]["sd"]["base_url"] == "http://127.0.0.1:1235"
    assert saved["backends"]["sd"]["launch"]["diffusion_model"] == str(model)
    assert "remote" not in saved["backends"]
    reloaded = load_settings(str(profile)).backends["sd"]
    assert reloaded.base_url == "http://127.0.0.1:1235"  # type: ignore[union-attr]
    assert reloaded.launch.executable == Path(sys.executable)  # type: ignore[union-attr]  # kept from the profile
    backends = {b["name"]: b for b in (await engines_client.get("/v1/backends")).json()}
    assert backends["sd"]["status"] == "down"  # the new endpoint is in use

    bad = await engines_client.patch("/v1/engines/sd", json={"launch": {"files": {"weights": "x"}}})
    assert bad.status_code == 422 and "weights" in bad.text
    remote = await engines_client.patch(
        "/v1/engines/remote", json={"launch": {"files": {"model": None}}}
    )
    assert remote.status_code == 200


async def test_start_and_stop_from_the_api(
    engines_client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = await engines_client.post("/v1/engines/sd/start")
    assert refused.status_code == 422 and "--diffusion-model" in refused.text
    model = _gguf(tmp_path / "qwen.gguf")
    await engines_client.patch(
        "/v1/engines/sd", json={"launch": {"files": {"diffusion_model": str(model)}}}
    )
    # The executable is Python here, so run a stand-in engine instead of sd-server's flags.
    script = "import time; print('listening', flush=True); time.sleep(60)"
    monkeypatch.setattr(launch, "command", lambda config, exe: [str(exe), "-c", script])

    started = await engines_client.post("/v1/engines/sd/start")
    assert started.status_code == 200, started.text
    assert started.json()["process"]["state"] == "running"
    again = await engines_client.post("/v1/engines/sd/start")
    assert again.status_code == 409
    for _ in range(200):
        lines = (await engines_client.get("/v1/engines/sd/log")).json()
        if any(line["text"] == "listening" for line in lines):
            break
        await asyncio.sleep(0.02)
    else:
        pytest.fail("no output from the engine")
    assert (await engines_client.get("/v1/engines/sd/log?errors=true")).json() == []
    stopped = (await engines_client.post("/v1/engines/sd/stop")).json()
    assert stopped["process"]["state"] == "stopped"
    log_file = await engines_client.get("/v1/engines/sd/log/file")
    assert log_file.status_code == 200 and "listening" in log_file.text

    remote = await engines_client.post("/v1/engines/remote/start")
    assert remote.status_code == 422 and "no launch settings" in remote.text


async def test_start_and_install_exclude_each_other(
    engines_client: httpx.AsyncClient, fake_provisioner: None
) -> None:
    body = {"accelerator": "cpu", "version": "slow"}
    installing = await engines_client.post("/v1/engines/sd/install", json=body)
    assert installing.status_code == 200, installing.text
    refused = await engines_client.post("/v1/engines/sd/start")
    assert refused.status_code == 409 and "install is running" in refused.text
    await engines_client.post("/v1/engines/sd/install/cancel")
    for _ in range(200):
        sd = (await engines_client.get("/v1/engines/sd")).json()
        if sd["install"]["state"] != "running":
            break
        await asyncio.sleep(0.02)
    after = await engines_client.post("/v1/engines/sd/start")
    assert after.status_code == 422  # no longer the install: now it's the missing model


async def test_remote_engines_keep_their_path_and_install_nothing_here(
    engines_client: httpx.AsyncClient,
) -> None:
    resp = await engines_client.patch("/v1/engines/remote", json={"host": "192.0.2.10"})
    assert resp.status_code == 200, resp.text
    remote = resp.json()
    assert remote["base_url"] == "http://192.0.2.10:9/inference/llama"
    assert (remote["loopback"], remote["runtime"], remote["install"]) == (False, None, None)
    for action in ("install/plan", "install"):
        refused = await engines_client.post(f"/v1/engines/remote/{action}", json={})
        assert refused.status_code == 422 and "runs on 192.0.2.10" in refused.text


@pytest.mark.parametrize(
    ("before", "host", "port", "after"),
    [
        ("http://127.0.0.1:8080", "10.0.0.2", 9000, "http://10.0.0.2:9000"),
        (
            "http://127.0.0.1:8080/inference/llama/",
            "gpu.lan",
            80,
            "http://gpu.lan:80/inference/llama/",
        ),
        ("https://me:pw@gpu.lan/sd?x=1", "::1", 443, "https://me:pw@[::1]:443/sd?x=1"),
    ],
)
def test_endpoint_edits_keep_the_rest_of_the_url(
    before: str, host: str, port: int, after: str
) -> None:
    assert service._with_endpoint(before, host, port) == after


async def test_concurrent_saves_keep_every_change(
    engines_client: httpx.AsyncClient, profile: Path, tmp_path: Path
) -> None:
    model = _gguf(tmp_path / "qwen.gguf")
    patches: list[dict[str, Any]] = [
        {"port": 1300},
        {"declared_capabilities": ["t2i", "edit"]},
        {"launch": {"files": {"diffusion_model": str(model)}}},
        {"launch": {"extra_args": ["--threads", "8"]}},
        {"launch": {"autostart": True}},
    ]
    done = await asyncio.gather(*(engines_client.patch("/v1/engines/sd", json=p) for p in patches))
    assert [r.status_code for r in done] == [200] * len(patches)

    sd = (await engines_client.get("/v1/engines/sd")).json()
    assert sd["base_url"] == "http://127.0.0.1:1300"
    assert sd["declared_capabilities"] == ["edit", "t2i"]
    assert (sd["extra_args"], sd["autostart"]) == (["--threads", "8"], True)
    assert {f["key"]: f["path"] for f in sd["files"]}["diffusion_model"] == str(model)
    saved = yaml.safe_load(local_overrides_path(profile).read_text(encoding="utf-8"))
    assert saved["backends"]["sd"]["base_url"] == "http://127.0.0.1:1300"
    assert saved["backends"]["sd"]["launch"]["autostart"] is True
    assert saved["backends"]["sd"]["launch"]["diffusion_model"] == str(model)
    assert list(profile.parent.glob("*.tmp")) == []


async def test_settings_without_a_profile_are_not_saved(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    config = sd_config("http://127.0.0.1:1", retry={"attempts": 1})
    app = create_app(make_settings(backends={"sd": config}))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch("/v1/engines/sd", json={"port": 2000})
            assert resp.json()["settings_file"] is None
            assert resp.json()["base_url"] == "http://127.0.0.1:2000"
