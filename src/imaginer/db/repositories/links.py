from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.db.models import Instance, InstanceLink, PairingCode


async def instance_id(session: AsyncSession, new_id: str, now: datetime) -> str:
    """This instance's id, made on first use."""
    found = (await session.execute(sa.select(Instance.id).limit(1))).scalar_one_or_none()
    if found is not None:
        return found
    session.add(Instance(id=new_id, created_at=now))
    await session.flush()
    return new_id


async def list_links(session: AsyncSession) -> list[InstanceLink]:
    stmt = sa.select(InstanceLink).order_by(InstanceLink.created_at)
    return list((await session.execute(stmt)).scalars())


async def get_link(session: AsyncSession, link_id: str) -> InstanceLink | None:
    return await session.get(InstanceLink, link_id)


async def by_peer(session: AsyncSession, peer_instance_id: str) -> InstanceLink | None:
    stmt = sa.select(InstanceLink).where(InstanceLink.peer_instance_id == peer_instance_id)
    return (await session.execute(stmt)).scalar_one_or_none()


async def names(session: AsyncSession) -> set[str]:
    return set((await session.execute(sa.select(InstanceLink.name))).scalars())


async def insert_link(session: AsyncSession, row: InstanceLink) -> None:
    session.add(row)
    await session.flush()


async def delete_link(session: AsyncSession, link_id: str) -> bool:
    stmt = sa.delete(InstanceLink).where(InstanceLink.id == link_id).returning(InstanceLink.id)
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def replace_code(session: AsyncSession, row: PairingCode) -> None:
    """Make ``row`` the only pairing code: a new code always cancels the previous one."""
    await session.execute(sa.delete(PairingCode))
    session.add(row)
    await session.flush()


async def active_code(session: AsyncSession, now: datetime) -> PairingCode | None:
    await session.execute(sa.delete(PairingCode).where(PairingCode.expires_at <= now))
    return (await session.execute(sa.select(PairingCode).limit(1))).scalar_one_or_none()


async def delete_codes(session: AsyncSession) -> None:
    await session.execute(sa.delete(PairingCode))
