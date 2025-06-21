import os
import time
import uuid
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def setup_run_folder():
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())
    job_run_id = f"{current_datetime}_{unique_id}"
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    plots_folder = os.path.join("runs", year, month, day, job_run_id)
    os.makedirs(plots_folder, exist_ok=True)
    log_path = os.path.join(plots_folder, "bms-ml-log.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode='a')
    logging.info(f"Created folder structure for job run ID: {job_run_id}")
    return plots_folder, job_run_id

def timing_decorator(func):
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
def train_failure_model(data_path, target_column="IsDead"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Failure Prediction Model Accuracy: {acc:.4f}")
    return model, acc

@timing_decorator
def predict_failing_cells(model, df):
    # Returns unique cell IDs predicted to fail
    predictions = model.predict(df.drop(columns=["IsDead"]))
    failure_cells = df.loc[predictions == 1, "Cell ID"].unique().tolist()
    logging.info(f"Cells predicted to fail: {failure_cells}")
    return failure_cells

@timing_decorator
def run_full_failure_prediction_pipeline(data_path):
    plots_folder, job_run_id = setup_run_folder()
    df = pd.read_csv(data_path)
    model, acc = train_failure_model(data_path)
    failing_cells = predict_failing_cells(model, df)
    return {
        "accuracy": acc,
        "failing_cells": failing_cells,
        "job_run_id": job_run_id,
        "plots_folder": os.path.abspath(plots_folder)
    }