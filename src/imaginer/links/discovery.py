"""Finding other instances on the network with mDNS / DNS-SD (``_imaginer._tcp``).

Each instance with linking on advertises its instance id, name, link API version and URL,
and browses for the others. Discovery only lists instances: linking still takes a pairing
code. It also keeps links working when an address changes: a linked instance that stops
answering where it was is looked for at the address it now advertises, and moved there if
the same instance (same id, same token) answers.
"""

import asyncio
import contextlib
import re
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from imaginer.core.config import is_loopback_host
from imaginer.core.logging import get_logger
from imaginer.domain import models as dto
from imaginer.links.state import PEER_API_VERSION

if TYPE_CHECKING:
    from imaginer.links.service import LinkService

log = get_logger(__name__)

SERVICE_TYPE = "_imaginer._tcp.local."
TTL_S = 300.0
RESOLVE_TIMEOUT_MS = 3000
_LABEL = re.compile(r"[^A-Za-z0-9-]+")


@dataclass(frozen=True)
class Sighting:
    instance: dto.DiscoveredInstance
    candidates: tuple[str, ...]  # URLs to try, best first
    service: str
    seen: float


def parse(
    properties: Mapping[bytes, bytes | None], addresses: list[str], port: int | None
) -> tuple[dto.DiscoveredInstance, tuple[str, ...]] | None:
    """An advertisement as an instance, with the URLs it may be reached at."""
    props = {
        k.decode(errors="replace"): (v or b"").decode(errors="replace")
        for k, v in properties.items()
    }
    instance_id = props.get("id")
    if not instance_id:
        return None
    candidates: list[str] = []
    advertised = props.get("url", "")
    if advertised.startswith(("http://", "https://")) and not is_loopback_host(
        urlsplit(advertised).hostname or ""
    ):
        candidates.append(advertised.rstrip("/"))
    if port:
        for address in addresses:
            if is_loopback_host(address):
                continue
            host = f"[{address}]" if ":" in address else address
            candidates.append(f"http://{host}:{port}")
    if not candidates:
        return None
    try:
        version: int | None = int(props.get("v", ""))
    except ValueError:
        version = None
    instance = dto.DiscoveredInstance(
        instance_id=instance_id,
        name=props.get("name") or instance_id[:8],
        url=candidates[0],
        api_version=version,
    )
    return instance, tuple(dict.fromkeys(candidates))


class Discovery:
    def __init__(self, links: "LinkService") -> None:
        self._links = links
        self._sightings: dict[str, Sighting] = {}
        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._info: AsyncServiceInfo | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        try:
            self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
            self._info = self._advertisement()
            await self._zc.async_register_service(self._info, allow_name_change=True)
            self._browser = AsyncServiceBrowser(
                self._zc.zeroconf, SERVICE_TYPE, handlers=[self._on_change]
            )
            log.info("links.discovery_started", service=self._info.name)
        except Exception as exc:  # no multicast here (container, VPN): linking still works
            log.warning("links.discovery_unavailable", error=str(exc))

    def instances(self) -> list[dto.DiscoveredInstance]:
        now = time.monotonic()
        self._sightings = {k: s for k, s in self._sightings.items() if now - s.seen < TTL_S}
        found = []
        for sighting in self._sightings.values():
            link = self._links.by_peer(sighting.instance.instance_id)
            found.append(
                sighting.instance.model_copy(update={"link_id": link.id if link else None})
            )
        return sorted(found, key=lambda i: i.name.lower())

    async def aclose(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._zc is None:
            return
        with contextlib.suppress(Exception):
            if self._browser is not None:
                await self._browser.async_cancel()
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()

    def _advertisement(self) -> AsyncServiceInfo:
        links = self._links
        host = _LABEL.sub("-", socket.gethostname()).strip("-") or "imaginer"
        label = _LABEL.sub("-", links.name).strip("-")[:40] or "imaginer"
        addresses = _local_addresses()
        return AsyncServiceInfo(
            SERVICE_TYPE,
            f"{label}-{links.instance_id[-6:]}.{SERVICE_TYPE}",
            port=links.port,
            properties={
                "id": links.instance_id,
                "name": links.name,
                "v": str(PEER_API_VERSION),
                "url": links.advertise_url,
            },
            server=f"{host.lower()}.local.",
            addresses=[socket.inet_aton(a) for a in addresses],
        )

    def _on_change(
        self,
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if self._info is not None and name == self._info.name:
            return
        if state_change is ServiceStateChange.Removed:
            self._sightings = {k: s for k, s in self._sightings.items() if s.service != name}
            return
        task = asyncio.ensure_future(self._resolve(name))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _resolve(self, name: str) -> None:
        if self._zc is None:
            return
        info = AsyncServiceInfo(SERVICE_TYPE, name)
        if not await info.async_request(self._zc.zeroconf, RESOLVE_TIMEOUT_MS):
            return
        parsed = parse(info.properties, info.parsed_addresses(), info.port)
        if parsed is None or parsed[0].instance_id == self._links.instance_id:
            return
        instance, candidates = parsed
        self._sightings[instance.instance_id] = Sighting(
            instance, candidates, name, time.monotonic()
        )
        await self._follow(instance.instance_id, candidates)

    async def _follow(self, instance_id: str, candidates: tuple[str, ...]) -> None:
        """Move a link that stopped answering to where its instance now is."""
        link = self._links.by_peer(instance_id)
        if link is None or link.status == "ok" or link.url in candidates:
            return
        for url in candidates:
            with contextlib.suppress(Exception):
                if await self._links.update_url(link.id, url):
                    return


def _local_addresses() -> list[str]:
    """This host's IPv4 addresses on the network, the one with the default route first."""
    found: list[str] = []
    with contextlib.suppress(OSError):
        found += socket.gethostbyname_ex(socket.gethostname())[2]
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # sends nothing: it only picks the outgoing interface
        found.insert(0, probe.getsockname()[0])
    return [a for a in dict.fromkeys(found) if not is_loopback_host(a)]
