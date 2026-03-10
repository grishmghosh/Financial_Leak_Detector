"""add vendor_name to transactions

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("transactions")]
    if "vendor_name" not in columns:
        op.add_column(
            "transactions",
            sa.Column("vendor_name", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("transactions", "vendor_name")
