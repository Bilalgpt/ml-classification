import os
from data import load_data, split_data
from model import train_model, save_model
from evaluate import evaluate_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, os.path.join(PROJECT_ROOT, "models", "classifier.joblib"))


if __name__ == "__main__":
    main()
