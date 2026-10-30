"""linked instances: this instance's id, its links, and the pairing code

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-29
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "instance",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_instance")),
    )
    op.create_table(
        "instance_link",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("peer_instance_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("peer_name", sa.String(length=64), nullable=False),
        sa.Column("url", sa.String(length=255), nullable=False),
        sa.Column("outbound_token", sa.String(length=128), nullable=False),
        sa.Column("inbound_token_sha256", sa.String(length=64), nullable=False),
        sa.Column("api_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_instance_link")),
        sa.UniqueConstraint("peer_instance_id", name=op.f("uq_instance_link_peer_instance_id")),
        sa.UniqueConstraint("name", name=op.f("uq_instance_link_name")),
        sa.UniqueConstraint(
            "inbound_token_sha256", name=op.f("uq_instance_link_inbound_token_sha256")
        ),
    )
    op.create_table(
        "pairing_code",
        sa.Column("code_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failed_attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.PrimaryKeyConstraint("code_sha256", name=op.f("pk_pairing_code")),
    )


def downgrade() -> None:
    op.drop_table("pairing_code")
    op.drop_table("instance_link")
    op.drop_table("instance")
