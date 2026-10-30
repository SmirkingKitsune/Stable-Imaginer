import asyncio
from collections import deque
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.config import RetryConfig, TimeoutConfig
from imaginer.domain.errors import EngineError, EngineTimeoutError, EngineUnavailableError
from tests.fakes.live import LiveServer, refusing_client


class Scripted:
    """Answers /r with queued status codes (then 200); /slow sleeps; /stream streams."""

    def __init__(self) -> None:
        self.statuses: deque[int] = deque()
        self.hits = 0
        self.last_auth: str | None = None

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/r", self._r, methods=["GET", "POST"]),
                Route("/slow", self._slow),
                Route("/stream", self._stream, methods=["POST"]),
            ]
        )

    async def _r(self, request: Request) -> Response:
        self.hits += 1
        self.last_auth = request.headers.get("authorization")
        status = self.statuses.popleft() if self.statuses else 200
        return JSONResponse({"hit": self.hits, "detail": f"status {status}"}, status_code=status)

    async def _slow(self, request: Request) -> Response:
        await asyncio.sleep(2)
        return JSONResponse({})

    async def _stream(self, request: Request) -> Response:
        self.hits += 1
        status = self.statuses.popleft() if self.statuses else 200
        if status != 200:
            return JSONResponse({"error": "busy"}, status_code=status)

        async def body() -> AsyncIterator[str]:
            for i in range(3):
                yield f"line {i}\n"

        return StreamingResponse(body(), media_type="text/plain")


@pytest.fixture(scope="module")
def _server() -> Iterator[tuple[Scripted, LiveServer]]:
    scripted = Scripted()
    with LiveServer(scripted.app()) as server:
        yield scripted, server


@pytest.fixture
def scripted(_server: tuple[Scripted, LiveServer]) -> Scripted:
    scripted, _ = _server
    scripted.statuses.clear()
    scripted.hits = 0
    return scripted


def _transport(
    url: str, attempts: int = 3, api_key: str | None = None, client: httpx.AsyncClient | None = None
) -> HttpTransport:
    return HttpTransport(
        name="engine",
        base_url=url,
        timeouts=TimeoutConfig(connect=1, read=5, write=5, pool=1),
        retry=RetryConfig(attempts=attempts, backoff_initial_s=0.001, backoff_max_s=0.002),
        api_key=api_key,
        client=client,
    )


@pytest.fixture
def url(_server: tuple[Scripted, LiveServer]) -> str:
    return _server[1].url


async def test_connection_refused_is_retried_then_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _transport("http://refused", attempts=3, client=refusing_client())
    backoffs: list[int] = []
    real = transport._backoff

    async def counting(attempt: int, path: str, error: object) -> None:
        backoffs.append(attempt)
        await real(attempt, path, error)  # type: ignore[arg-type]

    monkeypatch.setattr(transport, "_backoff", counting)
    with pytest.raises(EngineUnavailableError, match="unreachable"):
        await transport.post_json("/r", {"x": 1})  # POST too: nothing was sent
    assert backoffs == [1, 2]
    await transport.aclose()


async def test_503_is_retried_even_for_post(scripted: Scripted, url: str) -> None:
    scripted.statuses.extend([503, 503])
    transport = _transport(url)
    assert (await transport.post_json("/r", {}))["hit"] == 3
    await transport.aclose()


async def test_502_retried_for_get_but_not_post(scripted: Scripted, url: str) -> None:
    transport = _transport(url)
    scripted.statuses.append(502)
    with pytest.raises(EngineError) as info:
        await transport.post_json("/r", {})
    assert scripted.hits == 1
    assert info.value.extra["engine_status"] == 502
    scripted.statuses.append(502)
    assert (await transport.get_json("/r"))["hit"] == 3
    await transport.aclose()


async def test_client_errors_are_not_retried(scripted: Scripted, url: str) -> None:
    scripted.statuses.extend([400, 200])
    transport = _transport(url)
    with pytest.raises(EngineError, match=r"HTTP 400 for /r: .*status 400"):
        await transport.get_json("/r")
    assert scripted.hits == 1
    await transport.aclose()


async def test_exhausted_503_maps_to_unavailable(scripted: Scripted, url: str) -> None:
    scripted.statuses.extend([503, 503, 503])
    transport = _transport(url)
    with pytest.raises(EngineUnavailableError):
        await transport.get_json("/r")
    await transport.aclose()


async def test_read_timeout(url: str) -> None:
    transport = _transport(url)
    with pytest.raises(EngineTimeoutError, match="timed out"):
        await transport.get_json("/slow", timeout_s=0.2, attempts=1)
    await transport.aclose()


async def test_bearer_key_is_sent(scripted: Scripted, url: str) -> None:
    transport = _transport(url, api_key="sekret")
    await transport.get_json("/r")
    assert scripted.last_auth == "Bearer sekret"
    await transport.aclose()


async def test_stream_retries_503_before_headers(scripted: Scripted, url: str) -> None:
    scripted.statuses.append(503)
    transport = _transport(url)
    async with transport.stream("POST", "/stream", json={}) as response:
        lines = [line async for line in response.aiter_lines()]
    assert lines == ["line 0", "line 1", "line 2"]
    assert scripted.hits == 2
    await transport.aclose()


async def test_stream_error_status(scripted: Scripted, url: str) -> None:
    scripted.statuses.append(500)
    transport = _transport(url)
    with pytest.raises(EngineError, match="HTTP 500"):
        async with transport.stream("POST", "/stream", json={}):
            pass
    await transport.aclose()
