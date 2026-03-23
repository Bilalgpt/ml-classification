# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A machine learning classification project using Python and scikit-learn. The project trains and evaluates classification models on a built-in scikit-learn dataset (e.g., Iris, Digits, or Breast Cancer).

## Setup Commands

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python src/main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_model.py

# Run a single test by name
pytest tests/test_model.py::test_accuracy_above_threshold

# Lint and format
flake8 src/ tests/
black src/ tests/
```

## Project Structure

```
├── src/
│   ├── main.py          # Entry point — loads data, trains, evaluates
│   ├── data.py          # Dataset loading and train/test splitting
│   ├── model.py         # Model definition, training, and persistence
│   └── evaluate.py      # Metrics, confusion matrix, reporting
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_evaluate.py
├── notebooks/           # Exploratory analysis (not part of the pipeline)
├── models/              # Saved model artifacts (.joblib files)
├── requirements.txt
└── CLAUDE.md
```

## Coding Conventions

- **Data flow**: raw data → `data.py` → `model.py` → `evaluate.py`. Keep these stages cleanly separated; no model logic in `data.py` and no data loading in `model.py`.
- **No global state**: pass datasets, models, and config explicitly as function arguments.
- **Reproducibility**: always pass `random_state=42` to any scikit-learn estimator or split that accepts it.
- **Model persistence**: save/load models using `joblib` to/from the `models/` directory.
- **Metrics**: report at minimum accuracy, precision, recall, and F1-score. Use `sklearn.metrics.classification_report` as the standard output format.
- **Type hints**: annotate all function signatures. Use `numpy.ndarray` for arrays and `sklearn.base.BaseEstimator` (or a concrete type) for models.
- **Tests**: use `pytest`. Tests should not retrain models from scratch — use small synthetic datasets or fixtures instead.
- **Directory READMEs**: every directory must contain a `README.md` that explains the directory's purpose, lists its files and what each does, and describes how it relates to the rest of the project.
