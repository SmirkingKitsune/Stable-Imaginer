from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


class Database:
    """Async engine + session factory. Sessions are short-lived and opened per operation."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self.engine: AsyncEngine = create_async_engine(url, echo=echo)
        if self.engine.dialect.name == "sqlite":
            _configure_sqlite(self.engine)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    def session(self) -> AsyncSession:
        return self.sessionmaker()

    async def dispose(self) -> None:
        await self.engine.dispose()


def _configure_sqlite(engine: AsyncEngine) -> None:
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        # Take transaction control away from the driver so we can BEGIN IMMEDIATE below.
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")  # off by default in SQLite; cascades need it
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    @event.listens_for(sync_engine, "begin")
    def _on_begin(connection: Any) -> None:
        # Take the write lock up front. A deferred transaction that reads and then writes
        # fails with SQLITE_BUSY_SNAPSHOT under WAL instead of waiting on busy_timeout.
        connection.exec_driver_sql("BEGIN IMMEDIATE")
