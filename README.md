# BMS-Flux: Battery Data Synthetic Generation & Anomaly Detection Platform

## Overview

**BMS-Flux** is a modular, agent-compatible platform for end-to-end battery sensor data handling and analysis. It consists of three cooperative MCP (Modular Command Protocol) servers:

1. **Data Preprocessing MCP**  
   Robustly cleans, engineers features, imputes, and scales battery sensor CSV data for downstream use.

2. **SDV Synthetic Data MCP**  
   Provides synthetic tabular data generation, statistical evaluation, and visualization using the Synthetic Data Vault (SDV).

3. **Hybrid Anomaly Detection MCP**  
   Detects anomalies in battery data using a hybrid of classical ML (SVM, Isolation Forest) and deep learning (Variational Autoencoder), with advanced visualization.

---

## Architecture

```
Raw Battery Data (CSV)
        │
        ▼
[Preprocessing MCP] ──> [SDV MCP (Synthetic Data)] ──> [Evaluation/Visualization]
        │
        └────────────> [Anomaly Detection MCP] ──> [3D/Statistical Visualizations]
```

---

## Quickstart

### 1. Preprocess Your Data

```bash
cd load-preprocessdata-mcp
uv sync
uv run --with mcp server.py
```
Tools exposed:
- `preprocess_battery_data_tool(data_path, chunksize)`  
  Runs full cleaning, engineering, imputation, and scaling.

### 2. Generate Synthetic Data

```bash
cd data-generator-mcp
uv sync
uv run --with mcp server.py
```
Tools exposed:
- `sdv_generate(folder_name)`
- `sdv_evaluate(folder_name)`
- `sdv_visualize(folder_name, table_name, column_name)`

### 3. Hybrid Anomaly Detection

```bash
cd feature-one-anamoly-detection-mcp
uv sync
uv run --with mcp server.py
```
Tools exposed:
- `detect_anomalies(data_path)`
- `visualize_anomalies_3d(result_csv, ...)`

---

## Data Specifications

- Battery data CSV must include (at minimum):  
  `CellVoltage`, `CellTemperature`, `InstantaneousCurrent`, `AmbientTemperature`, `CellSpecificGravity`
- For SDV, a `metadata.json` describing all tables/columns is required in the data folder.
- All MCP tools return paths to output CSVs or PNGs for easy chaining.

---

## Developer Notes

- Each MCP server is self-contained with its own dependencies.  
- You may orchestrate chains of these MCPs in an agent workflow or CLI pipeline.
- All servers communicate over the MCP protocol for agent compatibility.

---

## Contribution

Contributions welcome! Fork and PR, or open issues for bugs/feature requests.

---

## License

MIT