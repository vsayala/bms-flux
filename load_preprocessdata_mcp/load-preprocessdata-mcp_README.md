# Battery Data Preprocessing MCP

This MCP server provides robust, modular preprocessing for battery sensor CSV data. It prepares your raw files for ML, anomaly detection, or synthetic data generation.

## Features

- **Outlier & error cleaning** for voltage, temperature, specific gravity
- **Feature engineering** (power, resistance, rolling/lags, polynomials, etc.)
- **Robust imputation** (MICE, then mean for leftovers)
- **Scaling** for ML-readiness
- Modular tool functions or full pipeline in one step

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

### Typical Workflow

1. Clean → 2. Engineer features → 3. Impute (MICE) → 4. Impute+scale (mean+scaler)
2. Or just use `preprocess_battery_data_tool` for all steps.

## Input Requirements

- CSV with:
  `CellVoltage`, `CellTemperature`, `InstantaneousCurrent`, `AmbientTemperature`, `CellSpecificGravity`

## Output

- Each tool returns a CSV path for seamless chaining.

## License

MIT
