import logging

import numpy as np

from app.ml.feature_engineering import extract_transaction_features
from app.ml.model import LeakDetectionModel

logger = logging.getLogger(__name__)


async def build_training_dataset(conn, org_id):
    logger.info("Building training dataset (org_id=%s)", org_id)

    rows = await conn.fetch(
        """
        SELECT
            voucher_number,
            org_id,
            amount,
            check_date,
            department,
            description,
            leak_probability
        FROM transactions
        WHERE org_id = $1
        """,
        org_id,
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
    logger.info("Built dataset with %d samples", len(X))
    return X
