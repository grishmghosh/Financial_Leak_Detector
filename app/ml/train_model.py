import asyncio
import logging
import os
import joblib
from pathlib import Path
from uuid import UUID

from app.db.connection import init_db, close_db, get_pool
from app.ml.dataset_builder import build_training_dataset
from app.ml.model import LeakDetectionModel

logger = logging.getLogger(__name__)


async def train_model(org_id: UUID):
    logger.info("Starting model training for org %s", org_id)
    await init_db()
    pool = get_pool()

    async with pool.acquire() as conn:
        X = await build_training_dataset(conn, org_id)

    if len(X) == 0:
        await close_db()
        raise ValueError("No training data available")

    model = LeakDetectionModel()
    model.model.fit(X)
    logger.info("Model trained on %d samples", len(X))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model.model, "models/isolation_forest.pkl")
    logger.info("Model saved to models/isolation_forest.pkl")

    await close_db()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raw_org_id = os.environ.get("ORG_ID")
    if not raw_org_id:
        raise SystemExit("ORG_ID environment variable is required")
    asyncio.run(train_model(UUID(raw_org_id)))
