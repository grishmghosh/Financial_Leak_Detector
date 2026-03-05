import asyncio
import joblib

from app.db.connection import get_pool
from app.ml.dataset_builder import build_training_dataset
from app.ml.model import LeakDetectionModel


async def train_model():
    pool = await get_pool()

    async with pool.acquire() as conn:
        X = await build_training_dataset(conn)

    if len(X) == 0:
        raise ValueError("No training data available")

    model = LeakDetectionModel()
    model.model.fit(X)

    joblib.dump(model.model, "models/isolation_forest.pkl")


if __name__ == "__main__":
    asyncio.run(train_model())
