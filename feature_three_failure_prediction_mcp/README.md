# Feature Three: Failure Prediction MCP

This MCP server predicts which battery cells are likely to die soon using a classification model (XGBoost).

## Usage

- Exposes tool: `predict_cell_failure(data_path: str)`
- Logs and all outputs are saved per-run in a unique folder.

**Input:**
- `data_path`: CSV file (must have `IsDead` column and field names as in data-generator-mcp)

**Output:**
- Dict with accuracy, failing cell IDs, job_run_id, etc.

---

## How to Run

```bash
uv run --with mcp server.py
```

---
