import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from utils.model_utils import save_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path


def setup_run_folder():
    """
    Sets up a unique directory for the current run to store outputs and logs.
    Returns the path to the folder and a unique job run ID.
    """
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())
    job_run_id = f"{current_datetime}_{unique_id}"
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    plots_folder = os.path.join("runs", year, month, day, job_run_id)
    os.makedirs(plots_folder, exist_ok=True)
    log_path = os.path.join(plots_folder, "bms-ml-log.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )
    logging.info(f"Created folder structure for job run ID: {job_run_id}")
    return plots_folder, job_run_id


def timing_decorator(func):
    """
    Decorator to time and log the execution of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started execution of '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in '{func.__name__}': {e}")
            raise e
        else:
            elapsed_time = time.time() - start_time
            logging.info(f"Completed '{func.__name__}' in {elapsed_time:.2f} seconds")
            return result

    return wrapper


@timing_decorator
def train_failure_model(data_path, target_column="Hybrid Anomalies"):
    """
    Trains a failure prediction model using XGBoost on the given data.
    Returns the trained model, accuracy, and model path.
    """
    df = pd.read_csv(data_path)
    # --- Robustness Patch: Check required columns ---
    if target_column not in df.columns:
        raise ValueError(f"Missing required target column: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- FIX: Select only numeric features for training ---
    X_numeric = X.select_dtypes(include=np.number)

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    # Save with versioning
    model_path, _ = save_model(model, "failure_xgb", "models")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Failure Prediction Model Accuracy: {acc:.4f}")
    return model, acc, model_path


@timing_decorator
def predict_failing_cells(model, df):
    """
    Predicts which cells are likely to fail using the trained model.
    Returns a list of failing cell IDs.
    """
    # --- Robustness Patch: Check required columns ---
    if "Hybrid Anomalies" not in df.columns:
        raise ValueError("Missing 'Hybrid Anomalies' column in data for prediction")
    if "Cell ID" not in df.columns:
        raise ValueError("Missing 'Cell ID' column in data for prediction")

    # --- FIX: Select only numeric features for prediction ---
    X = df.drop(columns=["Hybrid Anomalies"])
    X_numeric = X.select_dtypes(include=np.number)

    # Returns unique cell IDs predicted to fail
    predictions = model.predict(X_numeric)
    failure_cells = df.loc[predictions == 1, "Cell ID"].unique().tolist()
    logging.info(f"Cells predicted to fail: {failure_cells}")
    return failure_cells


@timing_decorator
def run_full_failure_prediction_pipeline(data_path):
    """
    Runs the full failure prediction pipeline: trains the model and predicts failing cells.
    Returns a dictionary with accuracy, failing cells, job run ID, plots folder, and model path.
    """
    plots_folder, job_run_id = setup_run_folder()
    df = pd.read_csv(data_path)
    model, acc, model_path = train_failure_model(data_path)
    failing_cells = predict_failing_cells(model, df)
    return {
        "accuracy": acc,
        "failing_cells": failing_cells,
        "job_run_id": job_run_id,
        "plots_folder": os.path.abspath(plots_folder),
        "model_path": model_path,
    }


def run_failure_prediction(data_path: str, run_folder: Path) -> dict:
    """
    Trains a RandomForest model to predict failures based on anomaly scores.
    Saves the model and classification report to a 'failure_prediction' folder.
    """
    failure_folder = run_folder / "failure_prediction"
    failure_folder.mkdir(exist_ok=True)
    logging.info(
        f"Starting failure prediction. Outputs will be saved to: {failure_folder}"
    )

    # This assumes the input data CSV now contains anomaly scores from the previous step
    df = pd.read_csv(data_path)

    # Define failure label based on anomaly detection results and other critical factors
    # CORRECTED: Use the 'anomaly' column from the new anomaly detection script
    df["failure_label"] = (
        (df["anomaly"] == -1) | (df["Cell Temperature (°C)"] > 75)
    ).astype(int)

    features = [
        "Voltage (V)",
        "Current (A)",
        "Cell Temperature (°C)",
        "SOC (%)",
        "anomaly_score",
    ]

    # Check for feature existence
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        msg = f"Failure prediction missing required features: {missing_features}"
        logging.error(msg)
        return {"error": msg}

    X = df[features]
    y = df["failure_label"]

    if X.empty or y.empty:
        raise ValueError("Empty DataFrame after feature selection")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    model_path = failure_folder / "failure_prediction_model.gz"
    joblib.dump(model, model_path)

    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)

    report_path = failure_folder / "classification_report.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)

    logging.info(f"Failure prediction training complete. Report saved to {report_path}")
    return {"failure_folder": str(failure_folder), "report_path": str(report_path)}
