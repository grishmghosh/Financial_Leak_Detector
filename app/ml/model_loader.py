import joblib
from pathlib import Path

from app.ml.model import LeakDetectionModel

MODEL_PATH = Path("models/isolation_forest.pkl")

_model_instance = None


def get_model():
    global _model_instance
    if _model_instance is not None:
        return _model_instance

    model = LeakDetectionModel()
    model.model = joblib.load(MODEL_PATH)
    _model_instance = model
    return _model_instance


def score_transaction(features: dict) -> float:
    model = get_model()
    return model.score(features)
