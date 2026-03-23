# src/

Source code for the ML classification pipeline. Files must be used in the order of the data flow: `data.py` → `model.py` → `evaluate.py`, orchestrated by `main.py`.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point. Orchestrates the full pipeline: load data, split, train, evaluate, save. |
| `data.py` | Loads the scikit-learn dataset and splits it into train/test sets. No model logic here. |
| `model.py` | Defines, trains, saves, and loads the classifier. No data loading here. |
| `evaluate.py` | Computes and prints metrics using `classification_report`. Returns a dict for use in tests. |

## Relation to the Rest of the Project

- `tests/` mirrors this directory with one test file per module.
- Trained models are persisted to `models/` via `model.save_model()`.
- `notebooks/` may import from this directory for interactive exploration but is not part of the pipeline.
