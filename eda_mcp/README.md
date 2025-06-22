# EDA MCP Module

This module performs robust, domain-aware exploratory data analysis (EDA) for battery management system (BMS) data, with a focus on cell-level analytics and industry-relevant diagnostics.

## Features
- **Cell Balancing Over Time:**
  - Plots and CSVs showing voltage (or SOC) spread across all cells at each time point.
  - Heatmaps of cell voltage/SOC by cell and time.
- **Failure Prediction by Cell:**
  - Per-cell failure/anomaly rates (if label exists).
  - Feature importance for predicting failure/anomaly.
- **Per-Cell and Per-Feature Analysis:**
  - Time series, boxplots, and outlier detection for each cell.
  - Comparison of key metrics (Voltage, Resistance, SOC, SOD, Temperature) across cells.
- **Categorical/Binary/Continuous Handling:**
  - Bar plots for categorical/binary features (e.g., charger status, alarms).
  - Histograms, KDE, and boxplots for continuous features.
- **More Domain Diagnostics:**
  - SOC/SOH drift detection per cell (plots, CSVs).
  - Voltage/temperature excursions per cell (plots, CSVs).
  - Cycle life/degradation analysis (capacity fade, resistance growth).
  - Alarm/event analysis (frequency, correlation with failures).
- **Interactive Dashboard:**
  - Tabs for summary, time series, boxplots, correlation heatmaps, and outlier highlighting.
  - Sidebar filters for cell selection, time range, and feature thresholds.
  - Export buttons for filtered data and plots as CSV/PNG.
  - Static EDA plots and diagnostics section.
  - Dashboard launches automatically after pipeline run (unless `--no-dashboard` is passed).
- **Logging:**
  - Every EDA step and error is logged to both a per-run `eda.log` and the pipeline-wide `bms_master.log`.
- **Outputs:**
  - All EDA outputs (plots, CSVs, reports) are saved in a unique run folder under `runs/YYYY/MM/DD/job_run_id/`.
  - Per-cell summary stats, cell balancing metrics, failure/anomaly rates by cell, outlier and drift detection results, domain diagnostics, dashboard.
  - HTML and PDF EDA reports with all outputs embedded or linked.

## Usage

Call the main EDA function with your data file:

```python
from eda_mcp.tools import run_eda
result = run_eda('data/input/your_data.csv')
print(result['run_folder'])  # Path to all outputs
```

### Launch the Interactive Dashboard

After running EDA, the dashboard will launch automatically (unless you pass `--no-dashboard` to the pipeline):

```bash
python agent-driver/run_full_pipeline.py
```

Or launch manually:

```bash
streamlit run eda_mcp/dashboard.py
```

## Output Structure
- `per_cell_summary.csv`: Per-cell summary statistics
- `cell_balancing_over_time.png`: Cell voltage/SOC over time
- `cell_voltage_heatmap.png`: Heatmap of cell voltages
- `failure_rate_by_cell.csv`/`.png`: Failure/anomaly rates by cell
- `feature_importance_failure.csv`/`.png`: Feature importance for failure/anomaly
- `bar_*.png`, `dist_*.png`: Plots for categorical/continuous features
- `cell_voltage_stats.csv`: Per-cell voltage stats
- `soc_drift_over_time.png`, `soh_drift_over_time.png`: SOC/SOH drift plots
- `soc_drift_stats.csv`, `soh_drift_stats.csv`: SOC/SOH drift stats
- `Voltage (V)_excursions.csv`/`.png`, `Temperature (C)_excursions.csv`/`.png`: Excursion events and plots
- `capacity_fade_vs_cycle.png`, `resistance_growth_vs_cycle.png`, etc.: Degradation plots
- `alarm_*_freq.png`: Alarm/event frequency plots
- `eda_report.html`/`.pdf`: Full EDA report
- `eda.log`: Per-run log
- `eda_outputs.txt`: List of all outputs

## Logging
- All steps and errors are logged to both `eda.log` (per run) and `bms_master.log` (pipeline-wide).

## Requirements
- See `pyproject.toml` for dependencies.
- For dashboard: `pip install streamlit plotly seaborn`

---
For questions or customization, see the code in `tools.py` or contact the maintainers. 