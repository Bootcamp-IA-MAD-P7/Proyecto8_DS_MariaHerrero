"""add assessment origin

Revision ID: c232d1c81201
Revises: c9c2ee2a986d
Create Date: 2026-08-28 13:16:51.860610

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c232d1c81201"

down_revision: Union[
    str,
    Sequence[str],
    None,
] = "c9c2ee2a986d"

branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None

depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


def upgrade() -> None:
    """Add assessment origin safely."""

    op.add_column(
        "assessments",
        sa.Column(
            "origin",
            sa.String(),
            nullable=False,
            server_default="self_reported",
        ),
    )

    with op.batch_alter_table(
        "assessments"
    ) as batch_op:
        batch_op.alter_column(
            "origin",
            server_default=None,
        )


def downgrade() -> None:
    """Remove assessment origin."""

    with op.batch_alter_table(
        "assessments"
    ) as batch_op:
        batch_op.drop_column(
            "origin"
        )