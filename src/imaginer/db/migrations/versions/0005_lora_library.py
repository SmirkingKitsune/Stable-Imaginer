"""the LoRA library: what the user adds to each LoRA, and the LoRA files' hashes

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "lora",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("default_weight", sa.Float(), nullable=True),
        sa.Column("tags", JSON, nullable=True),
        sa.Column("triggers", JSON, nullable=True),
        sa.Column("pinned", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("cover", JSON, nullable=True),
        sa.Column("compat", JSON, nullable=True),
        sa.Column("lineage", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_lora")),
    )
    op.create_table(
        "lora_file",
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("source", JSON, nullable=True),
        sa.Column("hashed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("path", name=op.f("pk_lora_file")),
    )
    op.create_index(op.f("ix_lora_file_sha256"), "lora_file", ["sha256"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_lora_file_sha256"), table_name="lora_file")
    op.drop_table("lora_file")
    op.drop_table("lora")
