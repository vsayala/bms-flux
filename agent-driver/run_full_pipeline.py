import sys
import os
import shutil
import logging
import time
import pandas as pd

# Ensure the project root is on the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from load_preprocessdata_mcp.tools import preprocess_battery_data
from feature_one_anamoly_detection_mcp.tools import run_hybrid_anomaly_detection
from feature_two_timeseries_prediction_mcp.tools import run_full_timeseries_pipeline
from feature_three_failure_prediction_mcp.tools import run_full_failure_prediction_pipeline

# Set up master BMS logger
logging.basicConfig(
    filename='bms_master.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode='a'
)
logger = logging.getLogger("bms_master")

def main():
    """
    Executes the full BMS-Flux pipeline by calling the tool functions directly.
    This is a simpler, more robust, and faster execution model than using MCP servers.
    """
    print("--- Starting Full BMS-Flux Pipeline ---")
    logger.info("--- Starting Full BMS-Flux Pipeline ---")

    # Clean up previous runs to ensure a fresh start
    if os.path.exists("runs"):
        shutil.rmtree("runs")
        print("Removed existing 'runs' directory.")
        logger.info("Removed existing 'runs' directory.")

    # Define data paths
    raw_data_path = "data/input/battery_data.csv"
    
    # Ensure output directory exists
    os.makedirs("data/output", exist_ok=True)

    # --- 1. Preprocessing ---
    print("\n[Step 1/5] Running Preprocessing...")
    logger.info("[Step 1/5] Running Preprocessing...")
    t0 = time.time()
    preprocessed_data_path = preprocess_battery_data(raw_data_path)
    t1 = time.time()
    print(f"Preprocessing complete. Output at: {preprocessed_data_path}")
    logger.info(f"Preprocessing complete. Output at: {preprocessed_data_path} (Elapsed: {t1-t0:.2f}s)")

    if not os.path.exists(preprocessed_data_path):
        print(f"Error: Preprocessed file not found at '{preprocessed_data_path}'. Aborting.")
        logger.error(f"Error: Preprocessed file not found at '{preprocessed_data_path}'. Aborting.")
        return

    # --- 2. Synthetic Data Generation ---
    print("\n[Step 2/5] Skipping Synthetic Data Generation (metadata.json not found)...")
    logger.info("[Step 2/5] Skipping Synthetic Data Generation (metadata.json not found)...")
    # sdv_generate("data/") 

    # --- 3. Anomaly Detection ---
    print("\n[Step 3/5] Running Anomaly Detection...")
    logger.info("[Step 3/5] Running Anomaly Detection...")
    t0 = time.time()
    anomaly_result_path, _ = run_hybrid_anomaly_detection(preprocessed_data_path)
    t1 = time.time()
    print(f"Anomaly detection complete. Results at: {anomaly_result_path}")
    logger.info(f"Anomaly detection complete. Results at: {anomaly_result_path} (Elapsed: {t1-t0:.2f}s)")
    
    # --- 4. Time Series Forecasting ---
    print("\n[Step 4/5] Running Time Series Forecasting...")
    logger.info("[Step 4/5] Running Time Series Forecasting...")
    t0 = time.time()
    forecast_result = run_full_timeseries_pipeline(preprocessed_data_path, cell_id=1, steps=10)
    t1 = time.time()
    print(f"Forecasting complete. Plot at: {forecast_result.get('plot_path')}")
    logger.info(f"Forecasting complete. Plot at: {forecast_result.get('plot_path')} (Elapsed: {t1-t0:.2f}s)")

    # --- 5. Failure Prediction ---
    print("\n[Step 5/5] Running Failure Prediction...")
    logger.info("[Step 5/5] Running Failure Prediction...")
    t0 = time.time()
    failure_result = run_full_failure_prediction_pipeline(anomaly_result_path)
    t1 = time.time()
    print(f"Failure prediction complete. Log folder at: {failure_result.get('plots_folder')}")
    logger.info(f"Failure prediction complete. Log folder at: {failure_result.get('plots_folder')} (Elapsed: {t1-t0:.2f}s)")

    print("\n--- Full BMS-Flux Pipeline Finished Successfully! ---")
    logger.info("--- Full BMS-Flux Pipeline Finished Successfully! ---")

    # --- Verification Step ---
    print("\n--- Verifying Data Preprocessing ---")
    logger.info("--- Verifying Data Preprocessing ---")
    print("\nOriginal Data:")
    raw_df = pd.read_csv(raw_data_path)
    raw_df.info()
    print(raw_df.head())

    print("\nPreprocessed Data:")
    preprocessed_df = pd.read_csv(preprocessed_data_path)
    preprocessed_df.info()
    print(preprocessed_df.head())

if __name__ == "__main__":
    main()