"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-10-28
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json() -> sa.types.TypeEngine[Any]:
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True, length=16)


def _id(name: str, *, nullable: bool) -> sa.Column[str]:
    return sa.Column(name, sa.String(length=36), nullable=nullable)


def _ts(name: str, nullable: bool = False) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "conversation",
        _id("id", nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        _ts("archived_at", nullable=True),
        _ts("deleted_at", nullable=True),
        sa.Column("metadata_json", _json(), nullable=True),
        sa.Column("last_seq", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_artifact_no", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation")),
    )
    op.create_index("ix_conversation_updated_at", "conversation", ["updated_at"])

    op.create_table(
        "message",
        _id("id", nullable=False),
        _id("conversation_id", nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", _enum("role", "system", "user", "assistant", "tool"), nullable=False),
        sa.Column(
            "status",
            _enum("status", "streaming", "complete", "error", "cancelled"),
            server_default="complete",
            nullable=False,
        ),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("reasoning_text", sa.Text(), nullable=True),
        sa.Column("tool_calls_json", _json(), nullable=True),
        sa.Column("tool_call_id", sa.String(length=128), nullable=True),
        sa.Column("backend", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("metadata_json", _json(), nullable=True),
        _ts("created_at"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_message_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_message")),
        sa.UniqueConstraint("conversation_id", "seq", name=op.f("uq_message_conversation_id_seq")),
    )

    op.create_table(
        "artifact",
        _id("id", nullable=False),
        _id("conversation_id", nullable=False),
        _id("message_id", nullable=True),
        sa.Column("kind", _enum("kind", "image", "audio", "video", "file", "text"), nullable=False),
        sa.Column(
            "source", _enum("source", "upload", "generated", "edited", "derived"), nullable=False
        ),
        sa.Column("mime", sa.String(length=127), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("has_alpha", sa.Boolean(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=False),
        sa.Column("metadata_json", _json(), nullable=True),
        _ts("created_at"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_artifact_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
            name=op.f("fk_artifact_message_id_message"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_artifact")),
        sa.UniqueConstraint(
            "conversation_id", "label", name=op.f("uq_artifact_conversation_id_label")
        ),
    )
    op.create_index("ix_artifact_conv", "artifact", ["conversation_id", "created_at"])
    op.create_index(
        "ix_artifact_conv_kind_source", "artifact", ["conversation_id", "kind", "source"]
    )

    op.create_table(
        "generation_job",
        _id("id", nullable=False),
        _id("conversation_id", nullable=False),
        _id("request_message_id", nullable=True),
        sa.Column("tool_call_id", sa.String(length=128), nullable=True),
        sa.Column("task", _enum("task", "t2i", "edit"), nullable=False),
        sa.Column("alpha", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("backend", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("params_json", _json(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.Column(
            "status",
            _enum("status", "queued", "running", "succeeded", "failed", "cancelled"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("engine_job_id", sa.String(length=128), nullable=True),
        sa.Column("progress_json", _json(), nullable=True),
        _ts("queued_at"),
        _ts("started_at", nullable=True),
        _ts("finished_at", nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversation.id"],
            name=op.f("fk_generation_job_conversation_id_conversation"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["request_message_id"],
            ["message.id"],
            name=op.f("fk_generation_job_request_message_id_message"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generation_job")),
    )
    op.create_index("ix_job_conv_status", "generation_job", ["conversation_id", "status"])
    op.create_index("ix_job_status", "generation_job", ["status"])

    op.create_table(
        "artifact_lineage",
        _id("child_id", nullable=False),
        _id("parent_id", nullable=False),
        sa.Column(
            "relation", _enum("relation", "edit_source", "reference", "mask"), nullable=False
        ),
        sa.Column("ordinal", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.CheckConstraint("child_id <> parent_id", name=op.f("ck_artifact_lineage_not_self")),
        sa.ForeignKeyConstraint(
            ["child_id"],
            ["artifact.id"],
            name=op.f("fk_artifact_lineage_child_id_artifact"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["artifact.id"],
            name=op.f("fk_artifact_lineage_parent_id_artifact"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "child_id", "parent_id", "relation", name=op.f("pk_artifact_lineage")
        ),
    )
    op.create_index("ix_artifact_lineage_parent", "artifact_lineage", ["parent_id"])

    op.create_table(
        "job_output",
        _id("job_id", nullable=False),
        _id("artifact_id", nullable=False),
        sa.Column("ordinal", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifact.id"],
            name=op.f("fk_job_output_artifact_id_artifact"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["generation_job.id"],
            name=op.f("fk_job_output_job_id_generation_job"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "artifact_id", name=op.f("pk_job_output")),
    )

    op.create_table(
        "message_attachment",
        _id("message_id", nullable=False),
        _id("artifact_id", nullable=False),
        sa.Column("ordinal", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifact.id"],
            name=op.f("fk_message_attachment_artifact_id_artifact"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["message.id"],
            name=op.f("fk_message_attachment_message_id_message"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", "artifact_id", name=op.f("pk_message_attachment")),
    )
    op.create_index("ix_message_attachment_artifact", "message_attachment", ["artifact_id"])


def downgrade() -> None:
    op.drop_table("message_attachment")
    op.drop_table("job_output")
    op.drop_table("artifact_lineage")
    op.drop_table("generation_job")
    op.drop_table("artifact")
    op.drop_table("message")
    op.drop_table("conversation")
