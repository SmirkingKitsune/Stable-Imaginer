"""Who is calling: the user (this instance's own UI or API clients) or a linked instance.

The user presents ``server.auth_token`` (nothing, when it is unset). A linked instance
presents the token this instance issued it when they paired. Routers then say who may
use them: ``require_user`` for administration, ``allow_gallery`` for conversations (as far
as ``links.gallery`` allows), ``require_link`` for the peer API.
"""

import hmac
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from imaginer.api.deps import ContainerDep
from imaginer.domain.errors import ForbiddenError, UnauthorizedError

_bearer = HTTPBearer(
    auto_error=False,
    description="Static token from IMAGINER_SERVER__AUTH_TOKEN (not required when unset), "
    "or a linked instance's token.",
)

_READ_METHODS = frozenset({"GET", "HEAD"})


@dataclass(frozen=True, slots=True)
class Principal:
    link_id: str | None = None  # None: the user

    @property
    def is_link(self) -> bool:
        return self.link_id is not None


USER = Principal()


async def require_auth(
    container: ContainerDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    expected = container.settings.server.auth_token
    presented = credentials.credentials if credentials is not None else None
    if (
        presented is not None
        and expected is not None
        and hmac.compare_digest(presented.encode(), expected.get_secret_value().encode())
    ):
        return USER
    if presented is not None:
        link_id = container.links.authenticate(presented)
        if link_id is not None:
            return Principal(link_id=link_id)
    if expected is None:
        return USER
    raise UnauthorizedError("missing or invalid bearer token")


PrincipalDep = Annotated[Principal, Depends(require_auth)]


async def require_user(principal: PrincipalDep) -> Principal:
    if principal.is_link:
        raise ForbiddenError("linked instances can't use this route")
    return principal


async def require_link(principal: PrincipalDep) -> Principal:
    if not principal.is_link:
        raise ForbiddenError("only linked instances use this route")
    return principal


async def allow_gallery(
    request: Request, principal: PrincipalDep, container: ContainerDep
) -> Principal:
    """The user always; a linked instance as far as this instance's links.gallery allows."""
    if not principal.is_link:
        return principal
    mode = container.settings.links.gallery
    if mode == "off":
        raise ForbiddenError("this instance doesn't share its conversations")
    if mode == "read" and request.method not in _READ_METHODS:
        raise ForbiddenError("this instance shares its conversations read-only")
    return principal
