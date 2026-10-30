"""ORM tables. Every artifact, message and job hangs off a conversation and cascades with it."""

from datetime import datetime
from enum import StrEnum
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from imaginer.db.base import Base
from imaginer.db.types import JSONType, UTCDateTime
from imaginer.domain.enums import (
    ArtifactKind,
    ArtifactSource,
    JobStatus,
    JobTask,
    LineageRelation,
    MessageStatus,
    Role,
)

_ID = sa.String(36)


def _enum[E: StrEnum](enum_cls: type[E], name: str) -> sa.Enum:
    """A VARCHAR + named CHECK constraint (no native enum type, so Postgres stays simple)."""
    return sa.Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=16,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
    )


def _fk(target: str, ondelete: str) -> sa.ForeignKey:
    return sa.ForeignKey(target, ondelete=ondelete)


class Conversation(Base):
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    title: Mapped[str | None] = mapped_column(sa.Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    archived_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    # Tombstone: set first when deleting, so a half-finished delete is never visible.
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    # Counters allocated with UPDATE ... RETURNING, race-free on SQLite and Postgres.
    last_seq: Mapped[int] = mapped_column(sa.Integer, default=0, server_default=sa.text("0"))
    last_artifact_no: Mapped[int] = mapped_column(
        sa.Integer, default=0, server_default=sa.text("0")
    )

    __table_args__ = (sa.Index("ix_conversation_updated_at", "updated_at"),)


class Message(Base):
    """A node of the conversation tree. ``parent_id`` is the message it follows (None for a
    first message); siblings are alternatives: an edited user message, or another attempt
    at a tool call. A branch is the path from a first message to a leaf."""

    __tablename__ = "message"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(_ID, _fk("conversation.id", "CASCADE"))
    parent_id: Mapped[str | None] = mapped_column(_ID, _fk("message.id", "CASCADE"), index=True)
    # Creation order across all branches; a message's ancestors always have lower seqs.
    seq: Mapped[int] = mapped_column(sa.Integer)
    role: Mapped[Role] = mapped_column(_enum(Role, "role"))
    status: Mapped[MessageStatus] = mapped_column(
        _enum(MessageStatus, "status"),
        default=MessageStatus.COMPLETE,
        server_default=MessageStatus.COMPLETE.value,
    )
    content_text: Mapped[str | None] = mapped_column(sa.Text)
    reasoning_text: Mapped[str | None] = mapped_column(sa.Text)
    tool_calls_json: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONType)
    tool_call_id: Mapped[str | None] = mapped_column(sa.String(128))
    backend: Mapped[str | None] = mapped_column(sa.String(64))
    model: Mapped[str | None] = mapped_column(sa.String(256))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    # The unique constraint's index also serves (conversation_id, seq) range scans.
    __table_args__ = (sa.UniqueConstraint("conversation_id", "seq"),)


class Artifact(Base):
    __tablename__ = "artifact"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(_ID, _fk("conversation.id", "CASCADE"))
    # The message that produced this artifact (upload's user message, or the tool message).
    message_id: Mapped[str | None] = mapped_column(_ID, _fk("message.id", "SET NULL"))
    kind: Mapped[ArtifactKind] = mapped_column(_enum(ArtifactKind, "kind"))
    source: Mapped[ArtifactSource] = mapped_column(_enum(ArtifactSource, "source"))
    mime: Mapped[str] = mapped_column(sa.String(127))
    storage_key: Mapped[str] = mapped_column(sa.String(255))
    sha256: Mapped[str] = mapped_column(sa.String(64))
    size_bytes: Mapped[int] = mapped_column(sa.BigInteger)
    width: Mapped[int | None] = mapped_column(sa.Integer)
    height: Mapped[int | None] = mapped_column(sa.Integer)
    has_alpha: Mapped[bool | None] = mapped_column(sa.Boolean)
    duration_ms: Mapped[int | None] = mapped_column(sa.Integer)
    label: Mapped[str] = mapped_column(sa.String(32))
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    __table_args__ = (
        sa.UniqueConstraint("conversation_id", "label"),
        sa.Index("ix_artifact_conv", "conversation_id", "created_at"),
        sa.Index("ix_artifact_conv_kind_source", "conversation_id", "kind", "source"),
    )


class MessageAttachment(Base):
    """Artifacts a message shows or attaches, in order. Provenance stays on artifact.message_id."""

    __tablename__ = "message_attachment"

    message_id: Mapped[str] = mapped_column(_ID, _fk("message.id", "CASCADE"), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(_ID, _fk("artifact.id", "CASCADE"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(sa.Integer, default=0, server_default=sa.text("0"))

    __table_args__ = (sa.Index("ix_message_attachment_artifact", "artifact_id"),)


class ArtifactLineage(Base):
    __tablename__ = "artifact_lineage"

    child_id: Mapped[str] = mapped_column(_ID, _fk("artifact.id", "CASCADE"), primary_key=True)
    parent_id: Mapped[str] = mapped_column(_ID, _fk("artifact.id", "CASCADE"), primary_key=True)
    relation: Mapped[LineageRelation] = mapped_column(
        _enum(LineageRelation, "relation"), primary_key=True
    )
    ordinal: Mapped[int] = mapped_column(sa.Integer, default=0, server_default=sa.text("0"))

    __table_args__ = (
        sa.CheckConstraint("child_id <> parent_id", name="not_self"),
        sa.Index("ix_artifact_lineage_parent", "parent_id"),
    )


class GenerationJob(Base):
    __tablename__ = "generation_job"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(_ID, _fk("conversation.id", "CASCADE"))
    request_message_id: Mapped[str | None] = mapped_column(_ID, _fk("message.id", "SET NULL"))
    tool_call_id: Mapped[str | None] = mapped_column(sa.String(128))
    task: Mapped[JobTask] = mapped_column(_enum(JobTask, "task"))
    # Transparent output. Prompt-driven in Qwen-Image-2.1 and valid for t2i and edit alike.
    alpha: Mapped[bool] = mapped_column(sa.Boolean, default=False, server_default=sa.false())
    backend: Mapped[str] = mapped_column(sa.String(64))
    model: Mapped[str] = mapped_column(sa.String(256))
    params_json: Mapped[dict[str, Any]] = mapped_column(JSONType)
    seed: Mapped[int | None] = mapped_column(sa.BigInteger)
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus, "status"))
    error: Mapped[str | None] = mapped_column(sa.Text)
    # The engine's own job id (sd-server native API), needed to cancel it remotely.
    engine_job_id: Mapped[str | None] = mapped_column(sa.String(128))
    progress_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    queued_at: Mapped[datetime] = mapped_column(UTCDateTime)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    __table_args__ = (
        sa.Index("ix_job_conv_status", "conversation_id", "status"),
        sa.Index("ix_job_status", "status"),
    )


class JobOutput(Base):
    __tablename__ = "job_output"

    job_id: Mapped[str] = mapped_column(_ID, _fk("generation_job.id", "CASCADE"), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(_ID, _fk("artifact.id", "CASCADE"), primary_key=True)
    ordinal: Mapped[int] = mapped_column(sa.Integer, default=0, server_default=sa.text("0"))
    # The exact seed of this output, so any single image can be regenerated.
    seed: Mapped[int | None] = mapped_column(sa.BigInteger)


# --- linked instances -----------------------------------------------------------------


class Instance(Base):
    """This installation's identity: one row, made on first start. Linked instances know
    it by this id, so a new address or name never makes it a stranger."""

    __tablename__ = "instance"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class InstanceLink(Base):
    """A paired instance. Each side issued the other a token: ours to call it is kept
    as is, the one it calls us with only as a hash."""

    __tablename__ = "instance_link"

    id: Mapped[str] = mapped_column(_ID, primary_key=True)
    peer_instance_id: Mapped[str] = mapped_column(_ID)
    name: Mapped[str] = mapped_column(sa.String(32))  # our name for it; linked backends use it
    peer_name: Mapped[str] = mapped_column(sa.String(64))  # what it calls itself
    url: Mapped[str] = mapped_column(sa.String(255))
    outbound_token: Mapped[str] = mapped_column(sa.String(128))
    inbound_token_sha256: Mapped[str] = mapped_column(sa.String(64))
    api_version: Mapped[int] = mapped_column(sa.Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    __table_args__ = (
        sa.UniqueConstraint("peer_instance_id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("inbound_token_sha256"),
    )


class PairingCode(Base):
    """The code another instance may pair with. Single use, short-lived, and dropped after
    a few wrong tries. Only its hash is stored."""

    __tablename__ = "pairing_code"

    code_sha256: Mapped[str] = mapped_column(sa.String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    failed_attempts: Mapped[int] = mapped_column(sa.Integer, default=0, server_default=sa.text("0"))
