import pandas as pd
import tempfile
import os
from feature_three_failure_prediction_mcp.tools import (
    run_full_failure_prediction_pipeline,
)


def test_run_full_failure_prediction_pipeline():
    df = pd.DataFrame(
        {
            "Cell ID": [1, 2, 3, 4],
            "Hybrid Anomalies": [0, 1, 0, 1],
            "Voltage (V)": [3.7, 3.8, 3.6, 3.9],
            "Current (A)": [0.5, 0.6, 0.4, 0.7],
            "Resistance (Ohms)": [0.1, 0.2, 0.15, 0.18],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)
        result = run_full_failure_prediction_pipeline(csv_path)
        assert "accuracy" in result
        assert "failing_cells" in result
