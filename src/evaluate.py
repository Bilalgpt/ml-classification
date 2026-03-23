import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report


def evaluate_model(model: BaseEstimator, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    return report
