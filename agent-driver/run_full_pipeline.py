import sys
import os
import shutil
import logging
import time
import pandas as pd
import subprocess
import uuid
from pathlib import Path

# Ensure the project root is on the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_preprocessdata_mcp.tools import preprocess_battery_data
from feature_one_anamoly_detection_mcp.tools import run_anomaly_detection
from feature_two_timeseries_prediction_mcp.tools import predict_cell_timeseries
from feature_three_failure_prediction_mcp.tools import run_failure_prediction
from eda_mcp.tools import run_eda

# Set up master BMS logger
logging.basicConfig(
    filename='bms_master.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='a'
)
logger = logging.getLogger("bms_master")

# Setup master log
MASTER_LOG = "bms_master.log"
def master_log(msg):
    with open(MASTER_LOG, "a") as f:
        f.write(f"[PIPELINE] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

def setup_run_folder():
    """Creates a single, unique run folder for the entire pipeline execution."""
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())
    job_run_id = f"{current_datetime}_{unique_id}"
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    run_folder = Path("runs") / year / month / day / job_run_id
    run_folder.mkdir(parents=True, exist_ok=True)
    return run_folder

def kill_process_on_port(port):
    """Finds and terminates any process running on the specified port."""
    try:
        if sys.platform == "win32":
            command = f"netstat -ano | findstr :{port}"
            process = subprocess.run(command, capture_output=True, text=True, shell=True)
            if process.stdout:
                pid = process.stdout.strip().split()[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True)
                logger.info(f"Killed process {pid} on port {port}")
        else:
            command = f"lsof -ti :{port}"
            process = subprocess.run(command, capture_output=True, text=True, shell=True)
            if process.stdout:
                pid = process.stdout.strip()
                subprocess.run(f"kill -9 {pid}", shell=True)
                logger.info(f"Killed process {pid} on port {port}")
    except Exception as e:
        logger.error(f"Failed to kill process on port {port}: {e}")

def main():
    """
    Executes the full BMS-Flux pipeline by calling the tool functions directly.
    This is a simpler, more robust, and faster execution model than using MCP servers.
    """
    logger.info("--- Starting Full BMS-Flux Pipeline ---")

    # Define data paths
    raw_data_path = "data/input/battery_data.csv"
    
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)

    # 1. Setup Single Run Folder for all outputs
    run_folder = setup_run_folder()
    logger.info(f"Created single run folder: {run_folder}")

    # 2. Preprocessing
    logger.info("[Step 1/5] Running Preprocessing...")
    t0 = time.time()
    preprocessed_path = preprocess_battery_data("data/input/battery_data.csv")
    logger.info(f"Preprocessing complete. Output: {preprocessed_path} (Elapsed: {time.time()-t0:.2f}s)")

    # 3. EDA
    logger.info("[Step 2/5] Running EDA...")
    t0 = time.time()
    eda_result = run_eda(preprocessed_path, run_folder)
    logger.info(f"EDA complete. Outputs at: {eda_result.get('eda_folder')} (Elapsed: {time.time()-t0:.2f}s)")
    
    # 4. Anomaly Detection
    logger.info("[Step 3/5] Running Anomaly Detection...")
    t0 = time.time()
    anomaly_result = run_anomaly_detection(preprocessed_path, run_folder)
    anomaly_results_path = anomaly_result.get('results_path') # Get the path to the results CSV
    logger.info(f"Anomaly Detection complete. Outputs at: {anomaly_result.get('anomaly_folder')} (Elapsed: {time.time()-t0:.2f}s)")
    
    # 5. Time Series Forecasting
    logger.info("[Step 4/5] Running Time Series Forecasting...")
    t0 = time.time()
    timeseries_result = predict_cell_timeseries(anomaly_results_path, "1", 100, run_folder) # Use anomaly results
    logger.info(f"Time Series Forecasting complete. Outputs at: {timeseries_result.get('timeseries_folder')} (Elapsed: {time.time()-t0:.2f}s)")

    # 6. Failure Prediction
    logger.info("[Step 5/5] Running Failure Prediction...")
    t0 = time.time()
    # Ensure failure prediction uses the output from anomaly detection
    failure_result = run_failure_prediction(anomaly_results_path, run_folder)
    logger.info(f"Failure Prediction complete. Outputs at: {failure_result.get('failure_folder')} (Elapsed: {time.time()-t0:.2f}s)")

    logger.info("--- Full BMS-Flux Pipeline Finished Successfully! ---")

    # --- Verification Step ---
    print("\n--- Verifying Data Preprocessing ---")
    logger.info("--- Verifying Data Preprocessing ---")
    print("\nOriginal Data:")
    raw_df = pd.read_csv(raw_data_path)
    raw_df.info()
    print(raw_df.head())

    print("\nPreprocessed Data:")
    preprocessed_df = pd.read_csv(preprocessed_path)
    preprocessed_df.info()
    print(preprocessed_df.head())

    # Launch dashboard unless --no-dashboard flag is passed
    if "--no-dashboard" not in sys.argv:
        DASHBOARD_PORT = 8501
        logger.info(f"Attempting to launch dashboard on port {DASHBOARD_PORT}...")
        kill_process_on_port(DASHBOARD_PORT)
        try:
            command = [
                sys.executable, "-m", "streamlit", "run", "eda_mcp/dashboard.py",
                "--server.port", str(DASHBOARD_PORT),
                "--server.headless", "true" # Prevents browser from opening automatically
            ]
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Dashboard launched. URL: http://localhost:{DASHBOARD_PORT}")
            print(f"\n✅ Dashboard is running at: http://localhost:{DASHBOARD_PORT}")
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
            print(f"❌ Failed to launch dashboard: {e}")

if __name__ == "__main__":
    main()