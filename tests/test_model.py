import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.tree import DecisionTreeClassifier

from model import train_model, save_model, load_model


def _make_dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.random((60, 4))
    y = np.repeat([0, 1, 2], 20)
    return X, y


def test_train_model_returns_fitted_estimator() -> None:
    X, y = _make_dataset()
    model = train_model(X, y)
    assert hasattr(model, "predict")
    assert model.predict(X[:1]).shape == (1,)


def test_accuracy_above_threshold() -> None:
    X, y = _make_dataset()
    model = train_model(X, y)
    preds = model.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy >= 0.7


def test_train_model_custom_hyperparameters() -> None:
    X, y = _make_dataset()
    model = train_model(X, y, n_estimators=10, max_depth=3)
    assert model.n_estimators == 10
    assert model.max_depth == 3


def test_train_model_custom_estimator() -> None:
    X, y = _make_dataset()
    estimator = DecisionTreeClassifier(random_state=42)
    model = train_model(X, y, estimator=estimator)
    assert model is estimator
    assert hasattr(model, "predict")
    assert model.predict(X[:1]).shape == (1,)


def test_save_and_load_model() -> None:
    X, y = _make_dataset()
    model = train_model(X, y)
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        save_model(model, path)
        loaded = load_model(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
    finally:
        os.unlink(path)
