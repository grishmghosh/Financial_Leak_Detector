"""create analysis tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-06
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── analysis_runs ────────────────────────────────────────
    op.create_table(
        "analysis_runs",
        sa.Column("id", UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.org_id"), nullable=False),
        sa.Column("triggered_by", UUID(), nullable=True),
        sa.Column("trigger_source", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), server_default="running", nullable=False),
        sa.Column("transactions_scored", sa.Integer(), nullable=True),
        sa.Column("high_risk_count", sa.Integer(), nullable=True),
        sa.Column("duplicates_found", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── analysis_results ─────────────────────────────────────
    op.create_table(
        "analysis_results",
        sa.Column("id", UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.org_id"), nullable=False),
        sa.Column("run_id", UUID(), sa.ForeignKey("analysis_runs.id"), nullable=False),
        sa.Column("voucher_number", sa.Text(), nullable=False),
        sa.Column("leak_probability", sa.Float(), nullable=True),
        sa.Column("risk_factors", ARRAY(sa.Text()), nullable=True),
        sa.Column("risk_category", sa.Text(), nullable=True),
        sa.Column("is_duplicate", sa.Boolean(), server_default="false"),
        sa.Column("duplicate_of", sa.Text(), nullable=True),
    )

    # ── vendor_risk_scores ───────────────────────────────────
    op.create_table(
        "vendor_risk_scores",
        sa.Column("id", UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.org_id"), nullable=False),
        sa.Column("run_id", UUID(), sa.ForeignKey("analysis_runs.id"), nullable=False),
        sa.Column("vendor_name", sa.Text(), nullable=False),
        sa.Column("total_spend", sa.Numeric(), nullable=True),
        sa.Column("transaction_count", sa.Integer(), nullable=True),
        sa.Column("anomaly_count", sa.Integer(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
    )

    # ── department_risk_scores ───────────────────────────────
    op.create_table(
        "department_risk_scores",
        sa.Column("id", UUID(), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("org_id", sa.Text(), sa.ForeignKey("organizations.org_id"), nullable=False),
        sa.Column("run_id", UUID(), sa.ForeignKey("analysis_runs.id"), nullable=False),
        sa.Column("department", sa.Text(), nullable=False),
        sa.Column("monthly_spend", sa.Numeric(), nullable=True),
        sa.Column("spike_score", sa.Float(), nullable=True),
        sa.Column("risk_category", sa.Text(), nullable=True),
    )

    # ── RLS policies ─────────────────────────────────────────
    for table in ("analysis_runs", "analysis_results", "vendor_risk_scores", "department_risk_scores"):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY {table}_org_isolation ON {table}
                USING (org_id = (
                    SELECT org_id FROM user_organizations
                    WHERE user_id = current_setting('app.current_user_id')::uuid
                    LIMIT 1
                ))
            """
        )


def downgrade() -> None:
    for table in ("department_risk_scores", "vendor_risk_scores", "analysis_results", "analysis_runs"):
        op.execute(f"DROP POLICY IF EXISTS {table}_org_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        op.drop_table(table)
