from datetime import UTC, datetime


def utcnow() -> datetime:
    """The current time as an aware UTC datetime. The only clock the app reads."""
    return datetime.now(UTC)
