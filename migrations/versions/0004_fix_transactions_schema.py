"""fix transactions schema

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("transactions")]

    # Rename department_name → department (only if department doesn't already exist)
    if "department_name" in columns and "department" not in columns:
        op.alter_column("transactions", "department_name", new_column_name="department")

    # Add description column if missing
    if "description" not in columns:
        op.add_column("transactions", sa.Column("description", sa.Text(), nullable=True))

    # Add leak_probability column if missing
    if "leak_probability" not in columns:
        op.add_column(
            "transactions",
            sa.Column("leak_probability", sa.Float(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("transactions")]

    if "leak_probability" in columns:
        op.drop_column("transactions", "leak_probability")

    if "description" in columns:
        op.drop_column("transactions", "description")

    if "department" in columns and "department_name" not in columns:
        op.alter_column("transactions", "department", new_column_name="department_name")
