# BMS-Flux

A robust, modular pipeline for battery management system (BMS) data analysis, including preprocessing, anomaly detection, time series forecasting, and failure prediction.

## Project Structure

- `agent-driver/` — Main pipeline script (`run_full_pipeline.py`)
- `feature_one_anamoly_detection_mcp/` — Anomaly detection module
- `feature_two_timeseries_prediction_mcp/` — Time series forecasting module
- `feature_three_failure_prediction_mcp/` — Failure prediction module
- `load_preprocessdata_mcp/` — Data preprocessing module
- `schema/` — Data schema definitions
- `utils/` — Shared utilities
- `data/input/` — Input data (not tracked in git)
- `data/output/` — All pipeline outputs (preprocessed data, results, etc.; not tracked in git)
- `models/` — Saved models (not tracked in git)
- `eda_mcp/` — EDA module (industry-standard exploratory data analysis)

## Usage

1. **Install dependencies:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Run the full pipeline:**
   ```sh
   python agent-driver/run_full_pipeline.py
   ```

3. **Outputs:**
   - Preprocessed data and results (e.g., `hybrid_results.csv`) are saved in the `data/output/` directory.
   - Logs are saved in the `runs/` directory.
   - Models are saved in the `models/` directory.

## Development Best Practices
- All code is modular and importable; no sys.path hacks.
- All temporary, output, and cache files are excluded via `.gitignore`.
- Data validation and type enforcement are performed during preprocessing.
- Add new features as submodules under the main package structure.

## Contributing
- Add unit tests for new features in a `tests/` directory.
- Update this README and `.gitignore` as needed.

## Environment Setup

1. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies (including dev tools):**
   ```sh
   pip install -e .[dev]
   ```

## Testing, Linting, Formatting, and Type Checking

- **Run all tests:**
  ```sh
  pytest
  ```
- **Lint code with ruff:**
  ```sh
  ruff .
  ```
- **Format code with black:**
  ```sh
  black .
  ```
- **Type check with mypy:**
  ```sh
  mypy .
  ```

## Pre-commit Hooks

Set up pre-commit to automatically check code quality before each commit:
```sh
pip install pre-commit
pre-commit install
```

## Configuration
- Place all input data in `data/input/`.
- All outputs are written to `data/output/`.
- You can add a `config.yaml` for pipeline parameters (see below for a sample).

## Reproducibility
- All random seeds are set in code where relevant for deterministic results.
- Document your environment and dependencies for full reproducibility.

## Contribution Guidelines
- Add unit tests for new features in `tests/`.
- Use type hints and docstrings for all public functions/classes.
- Run lint, format, and type check before pushing.
- Do not commit data, logs, models, or secrets.

## Sample config.yaml
```yaml
preprocessing:
  impute_strategy: mean
  scale_features: true
anomaly_detection:
  contamination: 0.05
  svm_nu: 0.05
  iforest_estimators: 500
```

---

*This project is now clean, robust, and ready for production or further research.*
