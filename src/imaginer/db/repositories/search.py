"""Text search over one page of conversations: titles, prompts, replies and labels.

Searching goes page by page in gallery order, so a client can show results while the rest
is still being scanned. Matching is a case-insensitive substring (LIKE), which needs no
index or extension and is quick for a single user's history; the database narrows each
page to candidate rows and the exact match and snippet are taken here.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from imaginer.db.models import Artifact, Conversation, Message
from imaginer.domain.enums import ArtifactKind, Role
from imaginer.domain.models import SearchField as Field

SNIPPET_CONTEXT = 32  # characters kept on each side of the match


@dataclass(frozen=True, slots=True)
class Match:
    field: Field
    snippet: str | None = None
    label: str | None = None  # the image whose prompt or label matched


async def count(session: AsyncSession, *, include_archived: bool) -> int:
    stmt = (
        sa.select(sa.func.count())
        .select_from(Conversation)
        .where(Conversation.deleted_at.is_(None))
    )
    if not include_archived:
        stmt = stmt.where(Conversation.archived_at.is_(None))
    return int((await session.execute(stmt)).scalar_one())


async def matches(
    session: AsyncSession,
    rows: Iterable[Conversation],
    query: str,
    fields: Iterable[Field],
) -> dict[str, Match]:
    """The best match in each conversation of ``rows``: title, then your prompts, then an
    image's prompt or label, then a reply."""
    wanted = set(fields)
    needle = query.casefold()
    rows = list(rows)
    ids = [r.id for r in rows]
    found: dict[str, Match] = {}

    def keep(conversation_id: str, match: Match) -> None:
        if conversation_id not in found:
            found[conversation_id] = match

    if "title" in wanted:
        for r in rows:
            if r.title and needle in r.title.casefold():
                keep(r.id, Match("title"))
    if not ids:
        return found
    pattern = f"%{_escape(query)}%"

    if "prompt" in wanted:
        stmt = (
            sa.select(Message.conversation_id, Message.content_text)
            .where(
                Message.conversation_id.in_(ids),
                Message.role == Role.USER,
                Message.content_text.ilike(pattern, escape="\\"),
            )
            .order_by(Message.seq.desc())
        )
        for cid, text in await session.execute(stmt):
            if (snippet := _snippet(text, needle)) is not None:
                keep(cid, Match("prompt", snippet))
        # An image's own prompt, as the tool was asked for it.
        image_prompts = (
            sa.select(Artifact.conversation_id, Artifact.label, Artifact.metadata_json)
            .where(
                Artifact.conversation_id.in_(ids),
                Artifact.kind == ArtifactKind.IMAGE,
                sa.cast(Artifact.metadata_json, sa.Text).ilike(pattern, escape="\\"),
            )
            .order_by(Artifact.id.desc())
        )
        for cid, label, metadata in await session.execute(image_prompts):
            prompt = (metadata or {}).get("prompt")
            if isinstance(prompt, str) and (snippet := _snippet(prompt, needle)) is not None:
                keep(cid, Match("prompt", snippet, label))

    if "label" in wanted:
        stmt = (
            sa.select(Artifact.conversation_id, Artifact.label)
            .where(
                Artifact.conversation_id.in_(ids),
                sa.func.lower(Artifact.label).like(f"{_escape(needle)}%", escape="\\"),
            )
            .order_by(Artifact.id)
        )
        for cid, label in await session.execute(stmt):
            keep(cid, Match("label", label=label))

    if "reply" in wanted:
        stmt = (
            sa.select(Message.conversation_id, Message.content_text)
            .where(
                Message.conversation_id.in_(ids),
                Message.role == Role.ASSISTANT,
                Message.content_text.ilike(pattern, escape="\\"),
            )
            .order_by(Message.seq.desc())
        )
        for cid, text in await session.execute(stmt):
            if (snippet := _snippet(text, needle)) is not None:
                keep(cid, Match("reply", snippet))
    return found


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _snippet(text: str | None, needle: str) -> str | None:
    """The match with some words around it, on one line: '…a tiny lighthouse glyph…'."""
    if not text:
        return None
    flat = " ".join(text.split())
    at = flat.casefold().find(needle)
    if at == -1:
        return None
    start = max(0, at - SNIPPET_CONTEXT)
    end = min(len(flat), at + len(needle) + SNIPPET_CONTEXT)
    if start > 0 and (space := flat.find(" ", start, at)) != -1:
        start = space + 1
    if end < len(flat) and (space := flat.rfind(" ", at + len(needle), end)) != -1:
        end = space
    return f"{'…' if start > 0 else ''}{flat[start:end]}{'…' if end < len(flat) else ''}"
