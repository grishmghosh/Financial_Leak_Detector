import asyncio
import numpy as np

from app.ml.feature_engineering import extract_transaction_features
from app.ml.model import LeakDetectionModel


async def build_training_dataset(conn):
    rows = await conn.fetch(
        """
        SELECT
            voucher_number,
            amount,
            check_date,
            department,
            description,
            leak_probability
        FROM transactions
        """
    )

    model = LeakDetectionModel()
    dataset = []

    for row in rows:
        transaction = type("Transaction", (), {
            "amount": row["amount"],
            "department": row["department"],
            "description": row["description"],
        })()

        features = extract_transaction_features(
            transaction,
            recent_transaction_count=0,
            department_avg_amount=0,
            department_std_amount=1,
        )

        vector = model.build_feature_vector(features)
        dataset.append(vector)

    X = np.array(dataset)
    return X
