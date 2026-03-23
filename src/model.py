import numpy as np
import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    estimator: BaseEstimator | None = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
) -> BaseEstimator:
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    estimator.fit(X_train, y_train)
    return estimator


def save_model(model: BaseEstimator, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> BaseEstimator:
    return joblib.load(path)
