 # Battery Data Preprocessing MCP

This MCP server provides robust, modular preprocessing for battery sensor CSV data. It prepares your raw files for ML, anomaly detection, or synthetic data generation.

## Features

- **Outlier & error cleaning** for voltage, temperature, specific gravity
- **Feature engineering** (power, resistance, rolling/lags, polynomials, etc.)
- **Robust imputation** (MICE, then mean for leftovers)
- **Scaling** for ML-readiness
- Modular tool functions or full pipeline in one step
- **Per-run logging and output folder with unique ID for traceability**

## Usage

### Install

```bash
uv sync
```

### Run Server

```bash
uv run --with mcp server.py
```

### Tool APIs

- `clean_and_filter_data(data_path, chunksize=100000)`
- `engineer_battery_features(csv_path)`
- `impute_battery_data_mice(csv_path)`
- `impute_and_scale_battery_data(csv_path)`
- `preprocess_battery_data_tool(data_path, chunksize=100000)`

Each function saves output in a unique timestamped folder (`runs/YYYY/MM/DD/job_run_id/`) with logs.

### Typical Workflow

Run each step individually, or just use `preprocess_battery_data_tool` for the full pipeline.

## Input Requirements

- CSV with:
  `CellVoltage`, `CellTemperature`, `InstantaneousCurrent`, `AmbientTemperature`, `CellSpecificGravity`

## Output

- Each tool returns a CSV path in a unique run folder for seamless chaining.
- Detailed logs for each run.

## License

MIT
