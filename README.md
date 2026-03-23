# ML Classification Project

A machine learning classification project using Python and scikit-learn. The project trains and evaluates classification models on a built-in scikit-learn dataset.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py
```

## Running Tests

```bash
pytest                                               # all tests
pytest tests/test_model.py                           # single file
pytest tests/test_model.py::test_accuracy_above_threshold  # single test
```

## Linting and Formatting

```bash
flake8 src/ tests/
black src/ tests/
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Source code for the ML pipeline |
| `tests/` | Pytest test suite |
| `notebooks/` | Exploratory analysis notebooks |
| `models/` | Saved model artifacts |
