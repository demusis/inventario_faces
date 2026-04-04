from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def as_utc(value: float | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=UTC)


def isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat()


def format_local_datetime(value: datetime | None) -> str:
    if value is None:
        return "-"
    local_value = value.astimezone()
    offset = local_value.strftime("%z")
    if offset:
        offset = f"UTC{offset[:3]}:{offset[3:]}"
    else:
        offset = "UTC"
    return f"{local_value.strftime('%Y-%m-%d %H:%M:%S')} (horário local {offset})"
