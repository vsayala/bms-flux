import pandas as pd
import numpy as np
import tempfile
import os
from load_preprocessdata_mcp.tools import preprocess_battery_data


def test_preprocess_battery_data():
    # Create a small sample dataframe
    df = pd.DataFrame(
        {
            "CellNumber": [1, 2],
            "PacketDateTime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "CellVoltage": [3.7, 3.8],
            "InstantaneousCurrent": [0.5, 0.6],
            "CellTemperature": [25, 26],
            "AmbientTemperature": [24, 25],
            "CellSpecificGravity": [1.2, 1.3],
            "SocLatestValueForEveryCycle": [90, 91],
            "DodLatestValueForEveryCycle": [10, 9],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test.csv")
        df.to_csv(csv_path, index=False)
        out_path = preprocess_battery_data(csv_path)
        result = pd.read_csv(out_path)
        # Check that required columns exist and types are correct
        assert "Cell ID" in result.columns
        assert result["Cell ID"].dtype == np.int64 or result["Cell ID"].dtype == int
        assert "Voltage (V)" in result.columns
        assert "Current (A)" in result.columns
        assert "SOC (%)" in result.columns
        assert not result.isnull().any().any()
