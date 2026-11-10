"""the model library: a card per image model, with its components and settings

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "image_model",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("family", sa.String(length=64), nullable=True),
        sa.Column("components", JSON, nullable=False),
        sa.Column("sources", JSON, nullable=True),
        sa.Column("launch", JSON, nullable=True),
        sa.Column("defaults", JSON, nullable=True),
        sa.Column("presets", JSON, nullable=True),
        sa.Column("prompt_hint", sa.Text(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", JSON, nullable=True),
        sa.Column("pinned", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("cover", JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_image_model")),
    )


def downgrade() -> None:
    op.drop_table("image_model")
