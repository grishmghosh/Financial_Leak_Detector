import logging

import joblib
from pathlib import Path

from app.ml.model import LeakDetectionModel

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/isolation_forest.pkl")

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Trained model not found. Run `python -m app.ml.train_model` first."
        )

    model = LeakDetectionModel()
    model.model = joblib.load(MODEL_PATH)
    _model_instance = model
    logger.info("Loaded ML model from %s", MODEL_PATH)
    return _model_instance


def score_transaction(features: dict) -> float:
    model = get_model()
    return model.score(features)
