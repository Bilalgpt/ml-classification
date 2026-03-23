import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def cross_validate_model(
    model: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 5
) -> np.ndarray:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def evaluate_model(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    return report
