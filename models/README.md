# models/

Saved model artifacts produced by the training pipeline. Files are written here by `src/model.save_model()` and read back by `src/model.load_model()`.

## Contents

| File pattern | Description |
|--------------|-------------|
| `*.joblib` | Serialised scikit-learn estimators saved with `joblib`. |

## Relation to the Rest of the Project

- Written by `src/main.py` at the end of a training run.
- Read by downstream inference code or notebooks that need a pre-trained model.
- Not committed to version control (add `models/*.joblib` to `.gitignore`).
