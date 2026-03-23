import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import load_data, split_data


def test_load_data_shapes() -> None:
    X, y = load_data()
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_split_data_proportions() -> None:
    X = np.zeros((100, 4))
    y = np.zeros(100)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20


def test_split_data_reproducible() -> None:
    X = np.random.rand(50, 4)
    y = np.zeros(50)
    split1 = split_data(X, y)
    split2 = split_data(X, y)
    np.testing.assert_array_equal(split1[0], split2[0])
