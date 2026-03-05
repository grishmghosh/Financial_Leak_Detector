from sklearn.ensemble import IsolationForest
import numpy as np


class LeakDetectionModel:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )

    def build_feature_vector(self, features: dict) -> list[float]:
        return [
            float(features["amount_log"]),
            float(int(features["is_large_transaction"])),
            float(int(features["is_procurement"])),
            float(int(features["description_contains_urgent"])),
            float(int(features["description_contains_manual"])),
            float(int(features["description_contains_adjustment"])),
            float(features["transaction_count_last_hour"]),
            float(features["amount_zscore"]),
        ]

    def score(self, features: dict) -> float:
        vector = self.build_feature_vector(features)
        raw_score = self.model.decision_function([vector])[0]
        risk = 1 - (raw_score + 1) / 2
        return float(np.clip(risk, 0.0, 0.99))
