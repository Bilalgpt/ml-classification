import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import train_model
from evaluate import evaluate_model


def _make_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.random((60, 4))
    y = np.repeat([0, 1, 2], 20)
    return X, y


def test_evaluate_returns_dict() -> None:
    X, y = _make_dataset()
    model = train_model(X, y)
    report = evaluate_model(model, X, y)
    assert isinstance(report, dict)
    assert "accuracy" in report


def test_evaluate_contains_per_class_metrics() -> None:
    X, y = _make_dataset()
    model = train_model(X, y)
    report = evaluate_model(model, X, y)
    for cls in ["0", "1", "2"]:
        assert cls in report
        assert "precision" in report[cls]
        assert "recall" in report[cls]
        assert "f1-score" in report[cls]
