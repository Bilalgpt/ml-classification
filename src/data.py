import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> tuple[np.ndarray, np.ndarray]:
    dataset = load_iris()
    return dataset.data, dataset.target


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
