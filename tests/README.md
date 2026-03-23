# tests/

Pytest test suite. One test file per `src/` module. Tests use small synthetic datasets or fixtures — they do not retrain on the full dataset.

## Files

| File | Purpose |
|------|---------|
| `test_data.py` | Tests for `src/data.py`: shape validation, split proportions, reproducibility. |
| `test_model.py` | Tests for `src/model.py`: fitting, accuracy threshold, save/load round-trip. |
| `test_evaluate.py` | Tests for `src/evaluate.py`: return type, presence of per-class metrics. |

## Relation to the Rest of the Project

Mirrors `src/` one-to-one. Each test file imports directly from its corresponding `src/` module via a `sys.path` insert. Tests never touch `models/` except through `tempfile` for save/load round-trips.
