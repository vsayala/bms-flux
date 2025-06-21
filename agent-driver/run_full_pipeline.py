import subprocess

def run_tool(cmd, description):
    print(f"Running: {description} ...")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error: {proc.stderr}")
        exit(1)
    print(proc.stdout)
    return proc.stdout

def main():
    # 1. Preprocessing
    run_tool("uv run --with mcp load-preprocessdata-mcp/server.py preprocess_battery_data_tool --data_path data/raw_battery.csv", "Preprocessing")
    # 2. Synthetic
    run_tool("uv run --with mcp data-generator-mcp/server.py sdv_generate --folder_name data/", "Synthetic Data Generation")
    # 3. Anomaly Detection
    run_tool("uv run --with mcp feature-one-anamoly-detection-mcp/feature-one-anamoly-detection-mcp_server.py detect_anomalies --data_path data/battery_preprocessed.csv", "Anomaly Detection")
    # 4. Forecast
    run_tool("uv run --with mcp feature-two-timeseries-prediction-mcp/server.py predict_cell_timeseries --data_path data/battery_preprocessed.csv --cell_id 1", "Forecasting")
    # 5. Failure Prediction
    run_tool("uv run --with mcp feature-three-failure-prediction-mcp/server.py predict_cell_failure --data_path data/battery_preprocessed.csv", "Failure Prediction")

if __name__ == "__main__":
    main()