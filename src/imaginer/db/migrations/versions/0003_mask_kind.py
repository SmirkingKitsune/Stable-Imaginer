"""artifact.kind 'mask': masks painted in the UI for edit_image (labels mask-1, mask-2, ...)

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-29
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD = ("image", "audio", "video", "file", "text")
_NEW = ("image", "mask", "audio", "video", "file", "text")


def _check(values: Sequence[str]) -> str:
    return "kind IN ({})".format(", ".join(f"'{v}'" for v in values))


def upgrade() -> None:
    with op.batch_alter_table("artifact") as batch:
        batch.drop_constraint(op.f("ck_artifact_kind"), type_="check")
        batch.create_check_constraint(op.f("ck_artifact_kind"), _check(_NEW))


def downgrade() -> None:
    op.execute("DELETE FROM artifact WHERE kind = 'mask'")
    with op.batch_alter_table("artifact") as batch:
        batch.drop_constraint(op.f("ck_artifact_kind"), type_="check")
        batch.create_check_constraint(op.f("ck_artifact_kind"), _check(_OLD))
