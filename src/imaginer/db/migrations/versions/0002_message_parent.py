"""message.parent_id: conversations become trees (branches)

Existing conversations are linear, so each message's parent is the message before it.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("message") as batch:
        batch.add_column(sa.Column("parent_id", sa.String(length=36), nullable=True))
        batch.create_foreign_key(
            op.f("fk_message_parent_id_message"),
            "message",
            ["parent_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.create_index("ix_message_parent_id", ["parent_id"])
    op.execute(
        "UPDATE message SET parent_id = ("
        " SELECT prev.id FROM message AS prev"
        " WHERE prev.conversation_id = message.conversation_id AND prev.seq < message.seq"
        " ORDER BY prev.seq DESC LIMIT 1)"
    )


def downgrade() -> None:
    with op.batch_alter_table("message") as batch:
        batch.drop_index("ix_message_parent_id")
        batch.drop_constraint(op.f("fk_message_parent_id_message"), type_="foreignkey")
        batch.drop_column("parent_id")
