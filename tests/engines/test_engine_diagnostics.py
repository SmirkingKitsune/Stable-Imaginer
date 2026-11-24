"""What a failed engine's process says, for the chat model (engines/diagnostics.py): how it
exited, the files it ran, the log lines that matter, and hints. The first test replays the
log of a real crash: Anima's diffusion model started with Qwen-Image-2.1's text encoder."""

import sys
from datetime import timedelta
from pathlib import Path

from imaginer.backends.registry import BackendRegistry
from imaginer.core.clock import utcnow
from imaginer.core.config import RoutingConfig
from imaginer.engines import diagnostics
from imaginer.engines.service import EngineService
from imaginer.engines.supervisor import EngineProcess, LogLine, Run
from imaginer.image_models import recipes
from tests.conftest import SettingsFactory, sd_config
from tests.fakes.live import dead_url

MODELS = Path("Z:/models") if sys.platform == "win32" else Path("/models")
ANIMA = (
    MODELS / "circlestone-labs/Anima/split_files/diffusion_models/anima-aesthetic-v1.0.safetensors"
)
QWEN_TE = MODELS / "Comfy-Org/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors"
# Anima's VAE is Qwen-Image's: right, though from a repo Anima's recipe doesn't name.
VAE = MODELS / "Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors"
ESC = "\x1b"
CRASH_LOG = [
    "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 32579 MiB):",
    f"[I] loading diffusion model from '{ANIMA}' --- diffusion_engine.cpp:697",
    f"[I] loading llm from '{QWEN_TE}' --- diffusion_engine.cpp:754",
    "[I] Version: Anima  --- diffusion_engine.cpp:982",
    "[I] listening on: http://127.0.0.1:11234 --- main.cpp:168",
    "[I] generate_image 1024x1024 --- image.cpp:1100",
    *[f"  |{'#' * n}{' ' * (50 - n)}|  {n}/50 - 2.9GB/s{ESC}[K" for n in (10, 30, 50)],
    "[I] get_learned_condition completed, taking 4.85s --- image.cpp:664",
    "[I] generating image: 1/1 - seed 1822174832 --- image.cpp:1005",
    r"D:\a\stable-diffusion.cpp\ggml\src\ggml.c:3378: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed",
]


def _crashed(lines: list[str], *, code: int = 3) -> EngineProcess:
    proc = EngineProcess("sd")
    argv = ["sd-server", "--diffusion-model", str(ANIMA), "--llm", str(QWEN_TE)]
    proc.run = Run(argv=[*argv, "--vae", str(VAE)], log_path=Path("sd.log"), started_at=utcnow())
    for text in lines:
        proc._line(text, None)
    proc.state = "exited"
    proc.run.exit_code, proc.run.exited_at = code, utcnow()
    return proc


def test_a_crash_says_why_and_points_at_the_file_that_doesnt_fit() -> None:
    report = diagnostics.report(
        _crashed(CRASH_LOG),
        family=recipes.family("anima"),
        models_dir=MODELS,
        since=utcnow() - timedelta(minutes=1),
    )
    assert report is not None and report.state == "exited"
    assert report.summary == "GGML_ASSERT(ggml_can_mul_mat(a, b)) failed"  # no source path
    assert report.exit == "exit code 3, aborted"
    details = report.details
    assert details["family"] == "Anima" and details["detected_model"] == "Anima"
    assert (
        details["files"]["llm"]
        == "Comfy-Org/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors"
    )
    # It failed sampling, before the VAE ran: of the files from unusual repos, only the
    # text encoder can be at fault.
    assert (
        details["failed_while"].startswith("sampling")
        and "VAE hadn't run" in details["failed_while"]
    )
    [unusual] = details["unusual_files"]
    assert unusual.startswith("llm (text encoder) Comfy-Org/Qwen-Image-2.1/")
    assert "circlestone-labs/Anima/split_files/text_encoders" in unusual
    hints = " ".join(details["hints"])
    assert "shape mismatch" in hints and "unusual files" in hints and "started again" in hints
    # The log as shown: no progress bars, terminal codes or source locations.
    tail = details["log_tail"]
    assert not any("|" in line or ESC in line or "---" in line for line in tail)
    assert tail[-2:] == ["[I] generating image: 1/1 - seed 1822174832", CRASH_LOG[-1]]


def test_a_failure_while_decoding_can_be_the_vaes() -> None:
    decoding = [*CRASH_LOG[:-1], "[I] sampling completed", "[I] decoding 1 latents", CRASH_LOG[-1]]
    report = diagnostics.report(
        _crashed(decoding), family=recipes.family("anima"), models_dir=MODELS, since=utcnow()
    )
    assert report is not None
    assert report.details["failed_while"] == "decoding the image (the VAE)"
    assert [u.split(" ")[0] for u in report.details["unusual_files"]] == ["llm", "vae"]


def test_exit_codes_are_named() -> None:
    assert diagnostics.exit_text(0xC0000409) == "exit code 0xC0000409, aborted at a failed check"
    assert diagnostics.exit_text(-9) == "exit code -9, killed, maybe for running out of memory"
    assert diagnostics.exit_text(7) == "exit code 7"
    assert diagnostics.exit_text(None) is None


def test_known_failures_get_hints() -> None:
    # Memory, not a GPU fault: CUDA reports running out as an error too.
    assert diagnostics.hints(["CUDA error: out of memory"]) == [diagnostics.HINTS[1][1]]
    assert "Memory ran out" in diagnostics.HINTS[1][1]
    assert "failed internal check" in diagnostics.hints(["GGML_ASSERT(x) failed"])[0]
    assert diagnostics.hints(["[I] all good"]) == []


def test_a_running_engine_shows_only_what_it_said_about_this_work() -> None:
    proc = EngineProcess("sd")
    since = utcnow()
    proc.run = Run(argv=["sd-server"], log_path=Path("sd.log"), started_at=since)
    proc.state = "running"
    proc.run.lines.append(LogLine(since - timedelta(seconds=5), "[E] an old failure", True))
    proc.run.lines.append(LogLine(since, "[E] failed to decode the latent", True))
    report = diagnostics.report(proc, family=None, models_dir=MODELS, since=since)
    assert report is not None and report.state == "running" and report.exit is None
    assert report.details["errors"] == ["[E] failed to decode the latent"]
    assert "started again" not in " ".join(report.details.get("hints", []))


def test_a_stopped_engine_says_to_start_it() -> None:
    proc = _crashed(["[I] listening"])
    proc.state = "stopped"
    report = diagnostics.report(proc, family=None, models_dir=MODELS, since=utcnow())
    assert report is not None and report.state == "stopped"
    assert report.details == {"state": "stopped", "hints": [diagnostics.STOPPED]}


async def test_an_engine_that_stopped_answering_is_seen_exiting(
    make_settings: SettingsFactory,
) -> None:
    """A real process that crashes a moment after the request found nothing listening: the
    report waits for its exit, so it has the exit code and the last line."""
    settings = make_settings(
        backends={"sd": sd_config(dead_url())}, routing=RoutingConfig(t2i="sd")
    )
    registry = BackendRegistry.from_settings(settings)
    engines = EngineService(settings, registry)
    script = (
        "import time; print('[I] generating image: 1/1', flush=True); time.sleep(0.3); "
        "print('ggml.c:3378: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed', flush=True); "
        "raise SystemExit(3)"
    )
    since = utcnow()
    await engines.supervisor.start("sd", [sys.executable, "-c", script])
    try:
        report = await engines.failure_report("sd", family=None, since=since, unreachable=True)
        assert report is not None and report.state == "exited"
        assert report.summary == "GGML_ASSERT(ggml_can_mul_mat(a, b)) failed"
        assert report.exit == "exit code 3, aborted"
        assert report.details["log_file"].endswith(".log")
        assert (
            await engines.failure_report("other", family=None, since=since, unreachable=False)
            is None
        )
    finally:
        await engines.supervisor.aclose()
        await registry.aclose()
