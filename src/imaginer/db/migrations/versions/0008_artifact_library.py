"""artifact library: pinned images, and artifacts found by their bytes

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "artifact_pin",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("sort_id", sa.String(length=36), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("mime", sa.String(length=127), nullable=False),
        sa.Column("storage_key", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("has_alpha", sa.Boolean(), nullable=True),
        sa.Column("label", sa.String(length=32), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("conversation_title", sa.Text(), nullable=True),
        sa.Column("snapshot", JSON, nullable=True),
        sa.Column("source_instance_id", sa.String(length=36), nullable=False),
        sa.Column("source_instance_name", sa.String(length=64), nullable=True),
        sa.Column("source_conversation_id", sa.String(length=36), nullable=True),
        sa.Column("source_artifact_id", sa.String(length=36), nullable=True),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", JSON, nullable=True),
        sa.Column("image_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_artifact_pin")),
        sa.UniqueConstraint("sha256", name=op.f("uq_artifact_pin_sha256")),
    )
    op.create_index("ix_artifact_pin_sort", "artifact_pin", ["sort_id"], unique=False)
    op.create_index("ix_artifact_sha256", "artifact", ["sha256"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_artifact_sha256", table_name="artifact")
    op.drop_index("ix_artifact_pin_sort", table_name="artifact_pin")
    op.drop_table("artifact_pin")
