"""Links to other Imaginer instances: pairing, the link table, and what each peer says.

Pairing (docs/linked-instances.md):

1. The responder mints a code (``mint_code``). It is shown there and typed in here.
2. The requester asks for a challenge (``GET /v1/link/hello``), then proves it knows the
   code (``POST /v1/link/pair``) and sends the token the responder will call it with.
3. The responder checks the proof, burns the code, and answers with the token the
   requester will call it with. Then it checks it can reach the requester back.

A wrong proof counts against the code; after a few, the code is gone. Each address may
only fail so often, too. Every comparison of secrets is constant-time, and only hashes of
issued tokens are stored.

Links are symmetric: once paired, each side polls the other's ``/v1/link/status`` for
what it shares, and may list its conversations. Nothing is transitive: an instance only
serves its own conversations and its own (non-linked) engines.
"""

import asyncio
import contextlib
import re
import socket
import time
from collections.abc import Coroutine
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from imaginer import __version__
from imaginer.backends.linked import linked_backend
from imaginer.backends.openai_http.transport import HttpTransport
from imaginer.core.clock import utcnow
from imaginer.core.config import (
    BackendConfig,
    LinkedBackendConfig,
    RetryConfig,
    Settings,
    TimeoutConfig,
)
from imaginer.core.ids import new_id
from imaginer.core.logging import get_logger
from imaginer.db import models as orm
from imaginer.db.engine import Database
from imaginer.db.repositories import links as links_repo
from imaginer.domain import models as dto
from imaginer.domain.errors import (
    ForbiddenError,
    ImaginerError,
    NotFoundError,
    PairingError,
    TooManyRequestsError,
)
from imaginer.links import pairing
from imaginer.links.state import PEER_API_VERSION, Link, LinkStatus

log = get_logger(__name__)

STATUS_TIMEOUT_S = 5.0
PAIR_TIMEOUT_S = 10.0
REVOKE_TIMEOUT_S = 3.0
MAX_CHALLENGES = 256
FAILURE_WINDOW_S = 600.0
MAX_FAILURES_PER_HOST = 10
RETURN_PATH_DELAYS_S = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)
# Peer requests stream for as long as an image takes; SSE pings keep the read alive.
_PEER_TIMEOUTS = TimeoutConfig(connect=5.0, read=300.0, write=60.0, pool=5.0)
_PEER_RETRY = RetryConfig(attempts=2, backoff_initial_s=0.5, backoff_max_s=2.0)
_NAME_CHARS = re.compile(r"[^a-z0-9_]+")


class LinkService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self.instance_id = ""
        self._links: dict[str, Link] = {}
        self._by_hash: dict[str, str] = {}  # sha256 of the token a peer calls us with -> link id
        self._outbound: dict[str, str] = {}  # link id -> the token we call the peer with
        self._challenges: dict[str, float] = {}  # challenge -> expiry (monotonic)
        self._failures: dict[str, list[float]] = {}  # address -> failed pairing attempts
        self._poll_locks: dict[str, asyncio.Lock] = {}
        self._retired: list[HttpTransport] = []  # replaced while a stream may still use it
        self._tasks: set[asyncio.Task[None]] = set()
        self.discovery: Any = None  # links.discovery.Discovery, when running

    # --- identity ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._settings.links.enabled

    @property
    def name(self) -> str:
        return self._settings.links.name or socket.gethostname()

    @property
    def port(self) -> int:
        return urlsplit(self.advertise_url).port or self._settings.server.port

    @property
    def advertise_url(self) -> str:
        configured = self._settings.links.advertise_url
        if configured:
            return configured
        host = self._settings.server.host
        if host in ("0.0.0.0", "::", ""):
            host = f"{socket.gethostname().lower()}.local"
        elif ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self._settings.server.port}"

    # --- lifecycle -----------------------------------------------------------------

    async def start(self, *, background: bool = True) -> None:
        """Load this instance's id and its links. With ``background``, also poll the links
        and run discovery (the server does; one-off CLI commands don't)."""
        async with self._db.session() as session, session.begin():
            self.instance_id = await links_repo.instance_id(session, new_id(), utcnow())
            rows = await links_repo.list_links(session)
        for row in rows:
            self._remember(row)
        if not self.enabled or not background:
            return
        self._spawn(self._poll_forever())
        if self._settings.links.discovery:
            from imaginer.links.discovery import Discovery

            self.discovery = Discovery(self)
            self._spawn(self.discovery.start())

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self.discovery is not None:
            await self.discovery.aclose()
        transports = [link.transport for link in self._links.values()] + self._retired
        await asyncio.gather(*(t.aclose() for t in transports), return_exceptions=True)

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- lookups -------------------------------------------------------------------

    def authenticate(self, token: str) -> str | None:
        """The link a peer's token belongs to, if any. Hashing first means the lookup
        can't leak anything about stored tokens through its timing."""
        if not self.enabled:
            return None
        return self._by_hash.get(pairing.token_hash(token))

    def all(self) -> list[Link]:
        return list(self._links.values())

    def get(self, link_id: str) -> Link:
        link = self._links.get(link_id)
        if link is None:
            raise NotFoundError(f"no link {link_id}")
        return link

    def by_name(self, name: str) -> Link | None:
        return next((link for link in self._links.values() if link.name == name), None)

    def by_peer(self, peer_instance_id: str) -> Link | None:
        return next(
            (link for link in self._links.values() if link.peer_instance_id == peer_instance_id),
            None,
        )

    def backend_factory(self, name: str, config: BackendConfig) -> Any:
        assert isinstance(config, LinkedBackendConfig)
        return linked_backend(name, config, self)

    # --- pairing: the side that shows the code ------------------------------------------

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise ForbiddenError("linking is off on this instance (links.enabled)")

    async def mint_code(self) -> dto.PairingCode:
        """A new pairing code; any earlier one stops working."""
        self._require_enabled()
        code = pairing.new_code()
        now = utcnow()
        expires = now + timedelta(seconds=pairing.CODE_TTL_S)
        row = orm.PairingCode(
            code_sha256=pairing.code_key(code), created_at=now, expires_at=expires
        )
        async with self._db.session() as session, session.begin():
            await links_repo.replace_code(session, row)
        log.info("link.code_minted", expires_at=expires.isoformat())
        return dto.PairingCode(code=code, expires_at=expires)

    def hello(self, nonce: str | None = None, asker: str | None = None) -> dto.LinkHello:
        """Who this is, and a pairing challenge. A linked instance (``asker``) also gets a
        proof that this is really the instance it linked with (see ``_proves_identity``)."""
        self._require_enabled()
        now = time.monotonic()
        self._challenges = {c: t for c, t in self._challenges.items() if t > now}
        while len(self._challenges) >= MAX_CHALLENGES:
            self._challenges.pop(next(iter(self._challenges)))
        challenge = pairing.new_challenge()
        self._challenges[challenge] = now + pairing.CHALLENGE_TTL_S
        proof = None
        link = self.by_peer(asker) if asker else None
        if nonce and link is not None:
            inbound = next((h for h, i in self._by_hash.items() if i == link.id), None)
            if inbound is not None:
                proof = pairing.identity_proof(inbound, nonce, self.instance_id)
        return dto.LinkHello(
            instance_id=self.instance_id,
            name=self.name,
            api_version=PEER_API_VERSION,
            version=__version__,
            challenge=challenge,
            proof=proof,
        )

    async def accept(self, body: dto.PairRequest, address: str | None) -> dto.PairAccepted:
        """Another instance pairs with this one."""
        self._require_enabled()
        self._check_failures(address)
        expiry = self._challenges.pop(body.challenge, None)
        if expiry is None or expiry < time.monotonic():
            self._failed(address)
            raise PairingError("unknown or expired challenge: start pairing again")
        if body.api_version != PEER_API_VERSION:
            raise PairingError(
                f"incompatible versions: this instance speaks link API {PEER_API_VERSION}, "
                f"the other {body.api_version}; update both"
            )
        if body.instance_id == self.instance_id:
            raise PairingError("an instance can't link with itself")

        token = pairing.new_token()
        now = utcnow()
        burned = False
        async with self._db.session() as session, session.begin():
            code = await links_repo.active_code(session, now)
            if code is None:
                accepted = False
            else:
                expected = pairing.proof(
                    code.code_sha256, body.challenge, body.instance_id, self.instance_id
                )
                accepted = pairing.proof_matches(expected, body.proof)
            if code is not None and not accepted:
                code.failed_attempts += 1
                if code.failed_attempts >= pairing.MAX_FAILED_ATTEMPTS:
                    await session.delete(code)
                    burned = True
            row: orm.InstanceLink | None = None
            if accepted:
                await links_repo.delete_codes(session)
                row = await self._store(session, body, token)
        if row is None:
            self._failed(address)
            if code is None:
                raise PairingError("no pairing code is active here: make one, then try again")
            hint = "; it is used up now, make a new one" if burned else ""
            raise PairingError(f"wrong pairing code{hint}")

        link = self._remember(row)
        log.info("link.accepted", link=link.name, peer=link.peer_name, url=link.url)
        self._spawn(self._confirm_return_path(link, address))
        return dto.PairAccepted(
            instance_id=self.instance_id, name=self.name, api_version=PEER_API_VERSION, token=token
        )

    async def _store(self, session: Any, body: dto.PairRequest, token: str) -> orm.InstanceLink:
        row = await links_repo.by_peer(session, body.instance_id)
        if row is None:
            row = orm.InstanceLink(
                id=new_id(),
                peer_instance_id=body.instance_id,
                name=_unique_name(body.name, await links_repo.names(session)),
                peer_name=body.name,
                url=body.url.rstrip("/"),
                outbound_token=body.token,
                inbound_token_sha256=pairing.token_hash(token),
                api_version=body.api_version,
                created_at=utcnow(),
            )
            await links_repo.insert_link(session, row)
        else:  # paired again (e.g. after the other side forgot the link): new tokens
            row.peer_name = body.name
            row.url = body.url.rstrip("/")
            row.outbound_token = body.token
            row.inbound_token_sha256 = pairing.token_hash(token)
            row.api_version = body.api_version
        return row

    async def _confirm_return_path(self, link: Link, address: str | None) -> None:
        """Check we can reach the instance that just paired. If its advertised URL doesn't
        answer, try the address its request came from, on the same port.

        The requester stores its side of the link only once it has our answer, so it may
        not know our token yet on the first tries: those are retried for a few seconds."""
        tried_address = False
        for delay in RETURN_PATH_DELAYS_S:
            await asyncio.sleep(delay)
            if link.id not in self._links:
                return  # unlinked meanwhile
            await self.poll(link)
            if link.status == "ok":
                return
            if link.status == "unreachable" and address and not tried_address:
                tried_address = True
                url = urlsplit(link.url)
                host = f"[{address}]" if ":" in address else address
                candidate = urlunsplit(url._replace(netloc=f"{host}:{url.port or 80}"))
                if candidate != link.url and await self.update_url(link.id, candidate):
                    return

    def _check_failures(self, address: str | None) -> None:
        key = address or "?"
        now = time.monotonic()
        recent = [t for t in self._failures.get(key, []) if now - t < FAILURE_WINDOW_S]
        self._failures[key] = recent
        if len(recent) >= MAX_FAILURES_PER_HOST:
            raise TooManyRequestsError("too many failed pairing attempts; try again later")

    def _failed(self, address: str | None) -> None:
        self._failures.setdefault(address or "?", []).append(time.monotonic())

    # --- pairing: the side that types the code --------------------------------------------

    async def pair_with(self, body: dto.LinkCreate) -> Link:
        """Pair with the instance at ``body.url`` using the code it shows."""
        self._require_enabled()
        url = body.url.rstrip("/")
        token = pairing.new_token()
        async with httpx.AsyncClient(base_url=url, timeout=PAIR_TIMEOUT_S) as client:
            hello = dto.LinkHello.model_validate(await _call(client, "GET", "/v1/link/hello"))
            if hello.api_version != PEER_API_VERSION:
                raise PairingError(
                    f"{hello.name} speaks link API {hello.api_version}, this instance "
                    f"{PEER_API_VERSION}; update both"
                )
            if hello.instance_id == self.instance_id:
                raise PairingError("that is this instance")
            request = dto.PairRequest(
                challenge=hello.challenge,
                proof=pairing.proof(
                    pairing.code_key(body.code),
                    hello.challenge,
                    self.instance_id,
                    hello.instance_id,
                ),
                instance_id=self.instance_id,
                name=self.name,
                url=self.advertise_url,
                api_version=PEER_API_VERSION,
                token=token,
            )
            answer = await _call(client, "POST", "/v1/link/pair", request.model_dump())
            accepted = dto.PairAccepted.model_validate(answer)
        if accepted.instance_id != hello.instance_id:
            raise PairingError("a different instance answered the pairing request")

        async with self._db.session() as session, session.begin():
            row = await links_repo.by_peer(session, accepted.instance_id)
            taken = await links_repo.names(session)
            if row is None:
                row = orm.InstanceLink(
                    id=new_id(),
                    peer_instance_id=accepted.instance_id,
                    name=body.name
                    if body.name and body.name not in taken
                    else _unique_name(body.name or accepted.name, taken),
                    peer_name=accepted.name,
                    url=url,
                    outbound_token=accepted.token,
                    inbound_token_sha256=pairing.token_hash(token),
                    api_version=accepted.api_version,
                    created_at=utcnow(),
                )
                await links_repo.insert_link(session, row)
            else:
                row.peer_name = accepted.name
                row.url = url
                row.outbound_token = accepted.token
                row.inbound_token_sha256 = pairing.token_hash(token)
                row.api_version = accepted.api_version
        link = self._remember(row)
        log.info("link.paired", link=link.name, peer=link.peer_name, url=link.url)
        await self.poll(link)
        return link

    async def unlink(self, link_id: str) -> None:
        """Forget the link here, and tell the peer to forget it too (best effort)."""
        link = self.get(link_id)
        with contextlib.suppress(ImaginerError):
            await link.transport.request(
                "POST",
                "/v1/link/revoke",
                timeout_s=REVOKE_TIMEOUT_S,
                attempts=1,
                accept_status=frozenset({401, 403, 404}),
            )
        await self.forget(link_id)

    async def forget(self, link_id: str) -> None:
        """Drop the link here only (the peer revoked it, or unlink already told it)."""
        async with self._db.session() as session, session.begin():
            await links_repo.delete_link(session, link_id)
        self._drop(link_id)

    def _drop(self, link_id: str) -> None:
        link = self._links.pop(link_id, None)
        if link is None:
            return
        self._by_hash = {h: i for h, i in self._by_hash.items() if i != link_id}
        self._outbound.pop(link_id, None)
        self._retired.append(link.transport)
        log.info("link.removed", link=link.name, peer=link.peer_name)

    # --- peer status ---------------------------------------------------------------------

    async def fresh(self, name: str, max_age_s: float) -> Link | None:
        link = self.by_name(name)
        if link is None:
            return None
        if link.checked_at is None or time.monotonic() - link.checked_at >= max_age_s:
            await self.poll(link)
        return link

    async def poll(self, link: Link) -> None:
        """Ask the peer what it shares. Concurrent callers share one request."""
        lock = self._poll_locks.setdefault(link.id, asyncio.Lock())
        if lock.locked():
            async with lock:
                return
        async with lock:
            before = link.status
            status, detail, peer = await self._status_of(link, link.transport)
            link.status, link.detail, link.checked_at = status, detail, time.monotonic()
            if peer is not None:
                link.peer, link.peer_name, link.seen_at = peer, peer.name, utcnow()
                link.engines = {e.name: e for e in peer.engines}
            if status != before:
                log.info("link.status", link=link.name, status=status, detail=detail)

    async def _status_of(
        self, link: Link, transport: HttpTransport
    ) -> tuple[LinkStatus, str | None, dto.PeerStatus | None]:
        try:
            response = await transport.request(
                "GET",
                "/v1/link/status",
                timeout_s=STATUS_TIMEOUT_S,
                attempts=1,
                accept_status=frozenset({401, 403, 404}),
            )
        except ImaginerError:
            return "unreachable", f"{link.peer_name} is unreachable at {transport.base_url}", None
        if response.status_code == 404:
            return "unauthorized", f"{link.peer_name} has linking turned off", None
        if response.status_code in (401, 403):
            return "unauthorized", f"{link.peer_name} no longer accepts this link: link again", None
        try:
            peer = dto.PeerStatus.model_validate(response.json())
        except (ValueError, ValidationError):
            return "incompatible", f"{link.peer_name} sent an unexpected status", None
        if peer.instance_id != link.peer_instance_id:
            return "incompatible", f"a different instance answers at {transport.base_url}", None
        if peer.api_version != PEER_API_VERSION:
            return (
                "incompatible",
                (
                    f"{peer.name} speaks link API {peer.api_version}, this instance "
                    f"{PEER_API_VERSION}; update both"
                ),
                None,
            )
        return "ok", None, peer

    async def sync(self) -> None:
        """Pick up links added or removed elsewhere (``imaginer link`` on this host)."""
        async with self._db.session() as session, session.begin():
            rows = {row.id: row for row in await links_repo.list_links(session)}
        for link_id in set(self._links) - set(rows):
            self._drop(link_id)
        for row in rows.values():
            link = self._links.get(row.id)
            known = link is not None and (
                link.url == row.url
                and self._outbound.get(row.id) == row.outbound_token
                and self._by_hash.get(row.inbound_token_sha256) == row.id
            )
            if not known:
                self._remember(row)

    async def _poll_forever(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                await self.sync()
            links = self.all()
            if links:
                await asyncio.gather(*(self.poll(link) for link in links), return_exceptions=True)
            await asyncio.sleep(self._settings.links.poll_interval_s)

    async def update_url(self, link_id: str, url: str) -> bool:
        """Move a link to a new address, if the same instance answers there."""
        link = self.get(link_id)
        url = url.rstrip("/")
        if url == link.url:
            return True
        if not await self._proves_identity(link, url):
            log.warning("link.move_refused", link=link.name, url=url)
            return False
        transport = _transport(link.name, url, self._outbound[link_id])
        status, _, _ = await self._status_of(link, transport)
        if status != "ok":
            await transport.aclose()
            return False
        async with self._db.session() as session, session.begin():
            row = await links_repo.get_link(session, link_id)
            if row is not None:
                row.url = url
        self._retired.append(link.transport)
        link.url, link.transport = url, transport
        log.info("link.moved", link=link.name, url=url)
        await self.poll(link)
        return True

    async def _proves_identity(self, link: Link, url: str) -> bool:
        """Whether the instance at ``url`` is the one we linked with. Asked before our token
        goes anywhere new: anyone on the network can claim an instance id (it is advertised),
        but only the linked instance holds the hash of the token we call it with."""
        nonce = pairing.new_challenge()
        params = {"nonce": nonce, "asker": self.instance_id}
        try:
            async with httpx.AsyncClient(base_url=url, timeout=STATUS_TIMEOUT_S) as client:
                response = await client.get("/v1/link/hello", params=params)
            hello = dto.LinkHello.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError):
            return False
        if hello.instance_id != link.peer_instance_id or hello.proof is None:
            return False
        token_hash = pairing.token_hash(self._outbound[link.id])
        expected = pairing.identity_proof(token_hash, nonce, link.peer_instance_id)
        return pairing.proof_matches(expected, hello.proof)

    # --- internals -------------------------------------------------------------------------

    def _remember(self, row: orm.InstanceLink) -> Link:
        old = self._links.get(row.id)
        if old is not None:
            self._retired.append(old.transport)
        self._by_hash = {h: i for h, i in self._by_hash.items() if i != row.id}
        link = Link(
            id=row.id,
            name=row.name,
            peer_instance_id=row.peer_instance_id,
            peer_name=row.peer_name,
            url=row.url,
            created_at=row.created_at,
            transport=_transport(row.name, row.url, row.outbound_token),
        )
        self._outbound[row.id] = row.outbound_token
        self._links[row.id] = link
        self._by_hash[row.inbound_token_sha256] = row.id
        return link


def _transport(name: str, url: str, token: str) -> HttpTransport:
    return HttpTransport(
        name=f"link {name}",
        base_url=url,
        timeouts=_PEER_TIMEOUTS,
        retry=_PEER_RETRY,
        api_key=token,
    )


async def _call(client: httpx.AsyncClient, method: str, path: str, json: Any = None) -> Any:
    """A pairing request; problems come back as PairingError with the peer's own words."""
    try:
        response = await client.request(method, path, json=json)
    except httpx.HTTPError as exc:
        raise PairingError(f"could not reach {client.base_url}: {exc!s}") from exc
    if response.status_code == 404:
        raise PairingError(f"{client.base_url} is not an Imaginer instance with linking on")
    if not response.is_success:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = None
        raise PairingError(detail or f"{client.base_url} answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise PairingError(f"{client.base_url} sent something that isn't JSON") from exc


def _unique_name(wanted: str, taken: set[str]) -> str:
    """A link name: lower-case letters, digits and _, starting with a letter, unused."""
    base = _NAME_CHARS.sub("_", wanted.lower()).strip("_") or "peer"
    if not base[0].isalpha():
        base = f"peer_{base}"
    base = base[:28]
    name, n = base, 2
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    return name
