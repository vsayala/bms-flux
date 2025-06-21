# Feature Two: Time Series Prediction MCP

This MCP server forecasts future battery cell parameters (voltage, current, temperature, etc.) for a given cell using XGBoost and lag features.

## Usage

- Exposes tool: `predict_cell_timeseries(data_path: str, cell_id: str, steps: int = 10)`
- Logs are saved per-run in a unique folder.

**Input:**  
- `data_path`: CSV file (must have field names as in data-generator-mcp)
- `cell_id`: Cell to forecast
- `steps`: How many future steps to predict

**Output:**  
- Dict with metrics, predictions, plot path, job_run_id, etc.

---

## How to Run

```bash
uv run --with mcp server.py
```

---