import pandas as pd
import tempfile
import os
from feature_one_anamoly_detection_mcp.tools import run_hybrid_anomaly_detection


def test_run_hybrid_anomaly_detection():
    df = pd.DataFrame(
        {
            "Voltage (V)": [3.7, 3.8, 3.6, 3.9, 3.7],
            "Cell Temperature (°C)": [25, 26, 24, 27, 25],
            "Current (A)": [0.5, 0.6, 0.4, 0.7, 0.5],
            "Ambient Temperature (°C)": [24, 25, 23, 26, 24],
            "CellSpecificGravity": [1.2, 1.3, 1.1, 1.4, 1.2],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)
        out_path, _ = run_hybrid_anomaly_detection(csv_path)
        result = pd.read_csv(out_path)
        assert "Hybrid Anomalies" in result.columns
        assert set(result["Hybrid Anomalies"].unique()).issubset({0, 1})
