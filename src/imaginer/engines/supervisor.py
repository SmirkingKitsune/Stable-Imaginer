"""Engine processes started from the Engines screen: start, stop, exit codes and logs.

Each run's output goes to a log file and to an in-memory tail, and is published to
subscribers (the log stream in the UI). Processes are plain ``subprocess.Popen`` with
reader threads, so this works on any asyncio loop on Windows and Linux. Stopping the
server stops the engines it started.
"""

import asyncio
import contextlib
import re
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from imaginer.core.clock import utcnow
from imaginer.core.logging import get_logger

log = get_logger(__name__)

State = Literal["stopped", "starting", "running", "stopping", "exited"]
TAIL_LINES = 2000
STOP_GRACE_S = 10.0
ERROR_RE = re.compile(r"\b(error|failed|fatal|exception|abort(ed)?|panic)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LogLine:
    at: datetime
    text: str
    error: bool


@dataclass
class Run:
    argv: list[str]
    log_path: Path
    started_at: datetime
    pid: int | None = None
    exited_at: datetime | None = None
    exit_code: int | None = None
    stopped_by_us: bool = False
    lines: deque[LogLine] = field(default_factory=lambda: deque(maxlen=TAIL_LINES))

    @property
    def last_error(self) -> str | None:
        for line in reversed(self.lines):
            if line.error:
                return line.text
        return None


Listener = Callable[[str, object], None]  # (event, payload): "line" LogLine | "state" State


class EngineProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.state: State = "stopped"
        self.run: Run | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._spawning: asyncio.Future[None] | None = None
        self._listeners: set[Listener] = set()
        self._exited = asyncio.Event()

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    def _emit(self, event: str, payload: object) -> None:
        for listener in list(self._listeners):
            try:
                listener(event, payload)
            except Exception:
                log.exception("engine.listener_failed", engine=self.name)

    def _set_state(self, state: State) -> None:
        self.state = state
        self._emit("state", state)

    @property
    def alive(self) -> bool:
        return self.state in ("starting", "running", "stopping")

    async def start(self, argv: list[str], log_path: Path) -> None:
        """Start the engine. If the caller is cancelled while the process is being created,
        it is still recorded once it exists, then ended: never left running unreachable."""
        if self.alive:
            raise RuntimeError(f"{self.name} is already running")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run = Run(argv=argv, log_path=log_path, started_at=utcnow())
        self._exited = asyncio.Event()
        self._set_state("starting")
        # The thread creates the process whatever happens to this caller meanwhile.
        spawn = asyncio.ensure_future(self._spawn(self.run))
        self._spawning = spawn
        try:
            await asyncio.shield(spawn)
        except asyncio.CancelledError:
            spawn.add_done_callback(self._abandon)
            raise

    def _abandon(self, spawn: asyncio.Future[None]) -> None:
        """A start whose caller gave up: kill the process it made, if it made one."""
        if spawn.cancelled() or spawn.exception() is not None:
            return
        if self._proc is None or self.run is None:  # it already exited
            return
        self.run.stopped_by_us = True
        self._set_state("stopping")
        with contextlib.suppress(OSError):
            self._proc.kill()

    async def _spawn(self, run: Run) -> None:
        loop = asyncio.get_running_loop()
        # A statement, not a conditional expression: mypy only narrows platform checks here.
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            flags = 0
        try:
            proc = await asyncio.to_thread(
                subprocess.Popen,
                run.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
            )
        except OSError as exc:
            self._line(f"could not start: {exc}", error=True)
            run.exit_code = -1
            run.exited_at = utcnow()
            self._set_state("exited")
            self._exited.set()
            raise
        self._proc = proc
        run.pid = proc.pid
        self._line(f"$ {' '.join(run.argv)}", error=False)
        self._set_state("running")
        log.info("engine.started", engine=self.name, pid=proc.pid)

        def pump() -> None:
            with run.log_path.open("a", encoding="utf-8") as sink:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    text = raw.rstrip("\r\n")
                    sink.write(raw if raw.endswith("\n") else raw + "\n")
                    sink.flush()
                    loop.call_soon_threadsafe(self._line, text, None)
            code = proc.wait()
            loop.call_soon_threadsafe(self._on_exit, run, code)

        threading.Thread(target=pump, name=f"engine-{self.name}", daemon=True).start()

    def _line(self, text: str, error: bool | None) -> None:
        if self.run is None:
            return
        line = LogLine(utcnow(), text, bool(ERROR_RE.search(text)) if error is None else error)
        self.run.lines.append(line)
        self._emit("line", line)

    def _on_exit(self, run: Run, code: int) -> None:
        run.exit_code = code
        run.exited_at = utcnow()
        if run is self.run:
            self._proc = None
            self._set_state("exited" if not run.stopped_by_us else "stopped")
            self._exited.set()
        log.info("engine.exited", engine=self.name, code=code, stopped=run.stopped_by_us)

    async def stop(self) -> None:
        spawning = self._spawning
        if spawning is not None and not spawning.done():
            # Still being created: there is no handle to stop it with until it exists.
            with contextlib.suppress(OSError):
                await asyncio.shield(spawning)
        proc = self._proc
        if proc is None or self.run is None:
            return
        self.run.stopped_by_us = True
        self._set_state("stopping")
        await asyncio.to_thread(proc.terminate)
        try:
            await asyncio.wait_for(self._exited.wait(), STOP_GRACE_S)
        except TimeoutError:
            with contextlib.suppress(OSError):
                await asyncio.to_thread(proc.kill)
            await self._exited.wait()


class Supervisor:
    def __init__(self, logs_root: Path) -> None:
        self._logs_root = logs_root
        self._procs: dict[str, EngineProcess] = {}

    def get(self, name: str) -> EngineProcess:
        if name not in self._procs:
            self._procs[name] = EngineProcess(name)
        return self._procs[name]

    def peek(self, name: str) -> EngineProcess | None:
        return self._procs.get(name)

    async def start(self, name: str, argv: list[str]) -> EngineProcess:
        proc = self.get(name)
        stamp = utcnow().strftime("%Y%m%d-%H%M%S")
        await proc.start(argv, self._logs_root / f"{name}-{stamp}.log")
        return proc

    async def stop(self, name: str) -> EngineProcess:
        proc = self.get(name)
        await proc.stop()
        return proc

    async def aclose(self) -> None:
        await asyncio.gather(
            *(p.stop() for p in self._procs.values() if p.alive), return_exceptions=True
        )
