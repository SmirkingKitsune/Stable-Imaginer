from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect

from imaginer.core.clock import utcnow

__all__ = ["JSONType", "UTCDateTime", "utcnow"]


class UTCDateTime(sa.types.TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend.

    SQLite has no timezone support: values are normalised to UTC on the way in and
    marked UTC on the way out, so callers only ever see aware datetimes.
    """

    impl = sa.DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed; use datetime.now(UTC)")
        return value.astimezone(UTC)

    def process_result_value(self, value: Any | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        assert isinstance(value, datetime)
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


# TEXT on SQLite, JSONB on Postgres.
JSONType = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
