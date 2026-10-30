import socket
import threading
import time
from types import TracebackType
from typing import Any, Literal

import httpx
import uvicorn


class LiveServer:
    """Serves an ASGI app on ``127.0.0.1:<free port>`` from a background thread.

    The app runs on that thread's own event loop, over real sockets, so streaming,
    disconnects and timeouts behave exactly as in production (on Windows too).
    """

    def __init__(self, app: Any, *, lifespan: Literal["auto", "on", "off"] = "off") -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, lifespan=lifespan, log_config=None, access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={"sockets": [self._sock]}, daemon=True
        )

    def start(self) -> "LiveServer":
        self._thread.start()
        deadline = time.monotonic() + 15
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"live server on {self.url} failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=15)
        self._sock.close()

    def __enter__(self) -> "LiveServer":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def dead_url() -> str:
    """A URL on which nothing listens (connection refused)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def refusing_client() -> httpx.AsyncClient:
    """An httpx client whose every request fails like a refused connection, instantly.

    Real refusals on Windows take ~2 s per attempt (the OS retries the SYN), so tests of
    retry policy use this; one test per adapter still exercises a real closed port.
    """

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("[Errno 111] Connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://refused")
