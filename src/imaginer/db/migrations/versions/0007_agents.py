"""agents: a chat model plus a persona, and the agent that wrote each message

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")


def upgrade() -> None:
    op.create_table(
        "agent",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("engine", sa.String(length=64), nullable=True),
        sa.Column("files", JSON, nullable=True),
        sa.Column("source", JSON, nullable=True),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("viewer", sa.String(length=16), server_default="reasoning", nullable=False),
        sa.Column("tools", JSON, nullable=True),
        sa.Column("sampling", JSON, nullable=True),
        sa.Column("context", sa.Integer(), nullable=True),
        sa.Column("portrait", JSON, nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("tags", JSON, nullable=True),
        sa.Column("pinned", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent")),
    )
    with op.batch_alter_table("message") as batch:
        batch.add_column(sa.Column("agent_id", sa.String(length=64), nullable=True))
        batch.create_index("ix_message_agent", ["agent_id"])


def downgrade() -> None:
    with op.batch_alter_table("message") as batch:
        batch.drop_index("ix_message_agent")
        batch.drop_column("agent_id")
    op.drop_table("agent")
