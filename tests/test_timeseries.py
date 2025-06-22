import pandas as pd
import numpy as np
import tempfile
import os
from feature_two_timeseries_prediction_mcp.tools import run_full_timeseries_pipeline

def test_run_full_timeseries_pipeline():
    df = pd.DataFrame({
        'Cell ID': [1]*10,
        'Timestamp': np.arange(10),
        'Voltage (V)': np.linspace(3.7, 3.8, 10),
        'Current (A)': np.linspace(0.5, 0.6, 10),
        'Resistance (Ohms)': np.linspace(0.1, 0.2, 10),
        'SOC (%)': np.linspace(90, 100, 10),
        'SOD (%)': np.linspace(10, 0, 10),
        'Cell Temperature (°C)': np.linspace(25, 30, 10),
        'Ambient Temperature (°C)': np.linspace(24, 29, 10),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'test.csv')
        df.to_csv(csv_path, index=False)
        result = run_full_timeseries_pipeline(csv_path, cell_id=1, steps=3)
        assert 'metrics' in result
        assert 'predictions' in result
        assert 'plot_path' in result 