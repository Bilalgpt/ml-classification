# notebooks/

Jupyter notebooks for exploratory data analysis and experimentation. Nothing in this directory is part of the production pipeline — `src/main.py` is the canonical entry point.

## Purpose

Use notebooks to:
- Inspect dataset distributions and class balance.
- Prototype and compare alternative models before promoting code to `src/`.
- Visualise confusion matrices or feature importances interactively.

## Relation to the Rest of the Project

Notebooks may import from `src/` for convenience but should not be imported by `src/`. Results or code worth keeping should be refactored into the appropriate `src/` module and covered by a test in `tests/`.
