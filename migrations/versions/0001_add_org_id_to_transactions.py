"""add org_id to transactions table

Revision ID: 0001
Revises: None
Create Date: 2026-03-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add org_id column to existing transactions table
    op.add_column(
        "transactions",
        sa.Column("org_id", sa.dialects.postgresql.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_transactions_org_id",
        "transactions",
        "organizations",
        ["org_id"],
        ["id"],
    )
    # Add created_at if it doesn't exist yet
    op.add_column(
        "transactions",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
    )

    # Enable RLS on transactions
    op.execute("ALTER TABLE transactions ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY transactions_org_isolation ON transactions
            USING (org_id = (
                SELECT org_id FROM user_organizations
                WHERE user_id = current_setting('app.current_user_id')::uuid
                LIMIT 1
            ))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS transactions_org_isolation ON transactions")
    op.execute("ALTER TABLE transactions DISABLE ROW LEVEL SECURITY")
    op.drop_constraint("fk_transactions_org_id", "transactions", type_="foreignkey")
    op.drop_column("transactions", "created_at")
    op.drop_column("transactions", "org_id")
