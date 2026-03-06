"""
ML orchestration service — batch analysis pipeline.
"""

import logging
from collections import defaultdict
from uuid import UUID

from app.db.connection import get_pool
from app.db.queries.ml_queries import (
    create_analysis_run,
    fetch_department_stats,
    fetch_transactions_for_analysis,
    insert_analysis_results,
    insert_department_risk_scores,
    insert_vendor_risk_scores,
    update_analysis_run,
    get_analysis_run,
)
from app.ml.duplicate_detector import detect_duplicates
from app.ml.scoring_engine import score_transaction
from app.config import get_settings
from app.schemas.ml import AnalysisRunRequest

logger = logging.getLogger(__name__)


def _risk_category(probability: float) -> str:
    settings = get_settings()
    if probability >= settings.HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= settings.MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


# ── Public API ──────────────────────────────────────────────

async def start_analysis(conn, *, org_id: UUID, user_id: UUID | None = None,
                         filters: AnalysisRunRequest | None = None) -> dict:
    """Create an analysis_runs record and return the run metadata."""
    run = await create_analysis_run(
        conn, org_id=org_id, triggered_by=user_id, trigger_source="api",
    )
    logger.info("Analysis run %s created for org %s", run["id"], org_id)
    return run


async def get_run_status(conn, run_id: UUID) -> dict | None:
    return await get_analysis_run(conn, run_id)


async def run_analysis_pipeline(
    run_id: UUID,
    org_id: UUID,
    filters: AnalysisRunRequest | None = None,
) -> None:
    """Execute the full batch analysis pipeline in the background."""
    pool = get_pool()

    logger.info("Pipeline started — run_id=%s org_id=%s", run_id, org_id)

    try:
        async with pool.acquire() as conn:
            # ── 1. Fetch transactions ────────────────────────
            txns = await fetch_transactions_for_analysis(
                conn, org_id,
                start_date=filters.start_date if filters else None,
                end_date=filters.end_date if filters else None,
                department=filters.department if filters else None,
            )

            if not txns:
                await update_analysis_run(conn, run_id=run_id, org_id=org_id, status="completed")
                logger.info("Pipeline finished — no transactions to process (run_id=%s)", run_id)
                return

            logger.info("Fetched %d transactions for scoring (run_id=%s)", len(txns), run_id)

            # ── 2. Department statistics for feature engineering ─
            dept_stats = await fetch_department_stats(conn, org_id)

            # ── 3. ML scoring ────────────────────────────────
            scored: list[dict] = []
            for txn in txns:
                dept = txn.get("department") or ""
                stats = dept_stats.get(dept, {"avg_amount": 0.0, "std_amount": 0.0, "txn_count": 0})

                txn_obj = type("Txn", (), {
                    "amount": txn["amount"],
                    "department": txn.get("department"),
                    "description": txn.get("description"),
                })()

                probability, risk_factors = score_transaction(
                    txn_obj,
                    recent_transaction_count=stats["txn_count"],
                    department_avg_amount=stats["avg_amount"],
                    department_std_amount=stats["std_amount"],
                )
                scored.append({
                    **txn,
                    "leak_probability": probability,
                    "risk_factors": risk_factors,
                    "risk_category": _risk_category(probability),
                })

            # ── 4. Duplicate detection ───────────────────────
            dup_map = detect_duplicates(txns)

            # ── 5. Build analysis_results rows ───────────────
            high_risk_count = 0
            duplicates_found = 0
            result_rows: list[dict] = []

            for s in scored:
                vn = s["voucher_number"]
                dup_info = dup_map.get(vn, {"is_duplicate": False, "duplicate_of": None})
                if dup_info["is_duplicate"]:
                    duplicates_found += 1
                if s["risk_category"] == "high":
                    high_risk_count += 1

                result_rows.append({
                    "org_id": org_id,
                    "run_id": run_id,
                    "voucher_number": vn,
                    "leak_probability": s["leak_probability"],
                    "risk_factors": s["risk_factors"],
                    "risk_category": s["risk_category"],
                    "is_duplicate": dup_info["is_duplicate"],
                    "duplicate_of": dup_info["duplicate_of"],
                })

            await insert_analysis_results(conn, result_rows)
            logger.info("Inserted %d analysis results (run_id=%s)", len(result_rows), run_id)

            # ── 6. Vendor risk aggregation ───────────────────
            vendor_agg: dict[str, dict] = defaultdict(lambda: {
                "total_spend": 0.0, "transaction_count": 0, "anomaly_count": 0,
            })
            for s in scored:
                vendor = (s.get("vendor_name") or s.get("description") or "unknown").strip()
                vendor_agg[vendor]["total_spend"] += float(s["amount"])
                vendor_agg[vendor]["transaction_count"] += 1
                if s["risk_category"] == "high":
                    vendor_agg[vendor]["anomaly_count"] += 1

            vendor_rows = []
            for vendor_name, agg in vendor_agg.items():
                ratio = agg["anomaly_count"] / agg["transaction_count"] if agg["transaction_count"] else 0.0
                vendor_rows.append({
                    "org_id": org_id,
                    "run_id": run_id,
                    "vendor_name": vendor_name,
                    "total_spend": agg["total_spend"],
                    "transaction_count": agg["transaction_count"],
                    "anomaly_count": agg["anomaly_count"],
                    "risk_score": round(ratio, 4),
                })
            await insert_vendor_risk_scores(conn, vendor_rows)
            logger.info("Inserted %d vendor risk scores (run_id=%s)", len(vendor_rows), run_id)

            # ── 7. Department risk aggregation ───────────────
            dept_agg: dict[str, dict] = defaultdict(lambda: {
                "monthly_spend": 0.0, "anomaly_count": 0, "count": 0,
            })
            for s in scored:
                dept = s.get("department") or "unknown"
                dept_agg[dept]["monthly_spend"] += float(s["amount"])
                dept_agg[dept]["count"] += 1
                if s["risk_category"] == "high":
                    dept_agg[dept]["anomaly_count"] += 1

            dept_rows = []
            for department, agg in dept_agg.items():
                stats = dept_stats.get(department, {"avg_amount": 0.0})
                avg = stats["avg_amount"] or 1.0
                spike_score = round(agg["monthly_spend"] / avg, 4) if avg else 0.0
                ratio = agg["anomaly_count"] / agg["count"] if agg["count"] else 0.0
                if ratio >= 0.3:
                    cat = "high"
                elif ratio >= 0.1:
                    cat = "medium"
                else:
                    cat = "low"
                dept_rows.append({
                    "org_id": org_id,
                    "run_id": run_id,
                    "department": department,
                    "monthly_spend": agg["monthly_spend"],
                    "spike_score": spike_score,
                    "risk_category": cat,
                })
            await insert_department_risk_scores(conn, dept_rows)
            logger.info("Inserted %d department risk scores (run_id=%s)", len(dept_rows), run_id)

            # ── 8. Finalize run ──────────────────────────────
            await update_analysis_run(
                conn,
                run_id=run_id,
                org_id=org_id,
                status="completed",
                transactions_scored=len(scored),
                high_risk_count=high_risk_count,
                duplicates_found=duplicates_found,
            )
            logger.info(
                "Pipeline completed — run_id=%s scored=%d high_risk=%d duplicates=%d",
                run_id, len(scored), high_risk_count, duplicates_found,
            )

    except Exception:
        logger.exception("Pipeline failed — run_id=%s", run_id)
        try:
            async with pool.acquire() as conn:
                await update_analysis_run(conn, run_id=run_id, org_id=org_id, status="failed")
        except Exception:
            logger.exception("Failed to update run status to 'failed' — run_id=%s", run_id)


async def retrain_model(conn, org_id: UUID) -> None:
    """Retrain the Isolation Forest model and reload it."""
    from app.ml.dataset_builder import build_training_dataset
    from app.ml.model import LeakDetectionModel
    from pathlib import Path
    import app.ml.model_loader as model_loader
    import joblib

    logger.info("Retraining ML model")

    X = await build_training_dataset(conn, org_id)
    if len(X) == 0:
        raise ValueError("No training data available for retraining")

    model = LeakDetectionModel()
    model.model.fit(X)
    logger.info("Model retrained on %d samples", len(X))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model.model, "models/isolation_forest.pkl")
    logger.info("Model saved to models/isolation_forest.pkl")

    # Force reload on next scoring call
    model_loader._model_instance = None
    model_loader.get_model()
    logger.info("Model reloaded into memory")
