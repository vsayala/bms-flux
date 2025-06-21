# BMS-Flux: Battery Data Synthetic Generation, Forecasting & Anomaly/Failure Detection Platform

## Overview

**BMS-Flux** is a modular, agent-compatible platform for end-to-end battery sensor data handling, synthetic generation, forecasting, anomaly, and failure detection. It uses the MCP (Modular Command Protocol) to enable agent-driven, notebook, or CLI-based orchestration of advanced ML pipelines.

The platform is organized into multiple MCP servers:

1. **Data Preprocessing MCP**  
   Cleans, engineers features, imputes, and scales battery sensor CSV data for any downstream use.

2. **SDV Synthetic Data MCP**  
   Provides synthetic tabular data generation, statistical evaluation, and visualization using the Synthetic Data Vault (SDV).

3. **Hybrid Anomaly Detection MCP**  
   Detects anomalies in battery data using a hybrid of classical ML (SVM, Isolation Forest) and deep learning (Variational Autoencoder), with advanced visualization.

4. **Time Series Forecasting MCP**  
   Forecasts future battery cell parameters (voltage, current, temperature, etc.) for a given cell using XGBoost and lag features.

5. **Failure Prediction MCP**  
   Predicts which battery cells are likely to die soon using a classification model (XGBoost), with per-run logging and traceability.

---

## Architecture

```
Raw Battery Data (CSV)
        │
        ▼
[Preprocessing MCP] ──> [SDV MCP (Synthetic Data)] ──> [Evaluation/Visualization]
        │
        ├────────────> [Anomaly Detection MCP] ──> [3D/Statistical Visualizations]
        │
        ├────────────> [Time Series Forecasting MCP] ──> [Parameter Forecast Plots, CSV]
        │
        └────────────> [Failure Prediction MCP] ──> [Failing Cell IDs, Reports]
```

---

## Quickstart

### 1. Preprocess Your Data

```bash
cd load-preprocessdata-mcp
uv sync
uv run --with mcp server.py
```
**Tools exposed:**
- `preprocess_battery_data_tool(data_path, chunksize)`  
  Full cleaning, feature engineering, imputation, and scaling (single command).
- Or use modular steps: `clean_and_filter_data`, `engineer_battery_features`, `impute_battery_data_mice`, `impute_and_scale_battery_data`

---

### 2. Generate Synthetic Data

```bash
cd data-generator-mcp
uv sync
uv run --with mcp server.py
```
**Tools exposed:**
- `sdv_generate(folder_name)`
- `sdv_evaluate(folder_name)`
- `sdv_visualize(folder_name, table_name, column_name)`

---

### 3. Hybrid Anomaly Detection

```bash
cd feature-one-anamoly-detection-mcp
uv sync
uv run --with mcp server.py
```
**Tools exposed:**
- `detect_anomalies(data_path)`
- `visualize_anomalies_3d(result_csv, ...)`

---

### 4. Time Series Forecasting (Voltage, Current, Temp, etc.)

```bash
cd feature-two-timeseries-prediction-mcp
uv sync
uv run --with mcp server.py
```
**Tools exposed:**
- `predict_cell_timeseries(data_path, cell_id, steps)`
  > Forecasts future values for several cell parameters, returns metrics, predictions, forecast plot.

---

### 5. Failure Prediction

```bash
cd feature-three-failure-prediction-mcp
uv sync
uv run --with mcp server.py
```
**Tools exposed:**
- `predict_cell_failure(data_path)`
  > Predicts which cells are likely to fail soon.

---

## Data Specifications

- Battery data CSV must include (at minimum):  
  `CellVoltage`, `CellTemperature`, `InstantaneousCurrent`, `AmbientTemperature`, `CellSpecificGravity`  
  (plus: `Cell ID`, `Timestamp` for forecasting/failure modules, and `IsDead` for failure labels)
- For SDV, a `metadata.json` describing all tables/columns is required in the data folder.
- All MCP tools return paths to output CSVs, PNGs, or JSONs for easy chaining.
- Each MCP server creates a unique run folder for logs and outputs.

---

...
## New Features

- **Model versioning:** All models are saved with timestamps and a latest symlink.
- **Standard output:** All tools return a dict with status, message, data, and log_path.
- **Health checks:** Each MCP exposes a health_check tool for agent orchestration.
- **Schema standardization:** All modules use a common schema (`schema/bms_schema.py`).
- **Live inference:** Real-time prediction API in `live-inference/`.
- **Pipeline automation:** Use `agent-driver/run_full_pipeline.py` for full workflow.
...

## Developer Notes

- Each MCP server is self-contained with its own dependencies and can be run independently.
- All intermediate and final outputs are logged and stored with a unique job run ID for traceability.
- You can orchestrate chains of these MCPs in an agent workflow, notebook, or CLI pipeline.
- All servers communicate over the MCP protocol for agent compatibility.

---

## Contribution

Contributions welcome! Fork and PR, or open issues for bugs/feature requests.

---

## License

MIT