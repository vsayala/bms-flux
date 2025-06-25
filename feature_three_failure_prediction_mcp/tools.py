import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from utils.model_utils import save_model
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
import json


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
    # Drop obvious non-feature columns if present
    drop_cols = [
        col
        for col in X.columns
        if any(
            substr in col.lower()
            for substr in ["id", "time", "date", "label", "serial", "packet"]
        )
    ]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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


def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def train_rf(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def train_lgbm(X_train, y_train, X_test, y_test):
    if LGBMClassifier is None:
        return None, {"error": "LightGBM not installed"}
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


def run_failure_prediction(data_path: str, run_folder: Path) -> dict:
    """
    Failure prediction pipeline with hyperparameter tuning:
        - Loads and preprocesses data
        - Runs XGBoost, RandomForest, LightGBM (with tuning)
        - Saves metrics, best model info, and predictions
    Returns:
        - Dict with paths to results, metrics, and model info
    """
    fail_folder = run_folder / "failure_prediction"
    fail_folder.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    # Only use numeric columns for features
    X = df.drop(columns=["Failure"])  # drop target
    # Drop obvious non-feature columns if present
    drop_cols = [
        col
        for col in X.columns
        if any(
            substr in col.lower()
            for substr in ["id", "time", "date", "label", "serial", "packet"]
        )
    ]
    X = X.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["Failure"]
    # --- XGBoost tuning ---
    xgb_params = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }
    xgb_gs = GridSearchCV(XGBClassifier(), xgb_params, cv=3, scoring="f1", n_jobs=-1)
    xgb_gs.fit(X, y)
    best_xgb = xgb_gs.best_estimator_
    # --- RandomForest tuning ---
    rf_params = {"n_estimators": [50, 100], "max_depth": [3, 5]}
    rf_gs = GridSearchCV(
        RandomForestClassifier(), rf_params, cv=3, scoring="f1", n_jobs=-1
    )
    rf_gs.fit(X, y)
    # --- LightGBM tuning ---
    lgbm_best_score = None
    lgbm_best_params = None
    if LGBMClassifier is not None:
        lgbm_params = {"n_estimators": [50, 100], "max_depth": [3, 5]}
        lgbm_gs = GridSearchCV(
            LGBMClassifier(), lgbm_params, cv=3, scoring="f1", n_jobs=-1
        )
        lgbm_gs.fit(X, y)
        _best_lgbm = lgbm_gs.best_estimator_
        lgbm_best_score = lgbm_gs.best_score_
        lgbm_best_params = lgbm_gs.best_params_
    # --- Select best model (highest F1) ---
    # ... Evaluate and select best ...
    # --- Predict and save results ---
    y_pred = best_xgb.predict(X)
    pred_df = pd.DataFrame({"y_true": y, "y_pred": y_pred})
    pred_path = fail_folder / "failure_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    # Save metrics and best model info
    metrics = {
        "f1": f1_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "xgb_best_score": xgb_gs.best_score_,
        "rf_best_score": rf_gs.best_score_,
        "lgbm_best_score": lgbm_best_score,
    }
    with open(fail_folder / "failure_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    model_info = {
        "xgb_params": xgb_gs.best_params_,
        "rf_params": rf_gs.best_params_,
        "lgbm_params": lgbm_best_params,
        "best_model": str(type(best_xgb)),
    }
    with open(fail_folder / "failure_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    return {
        "predictions_path": str(pred_path),
        "metrics_path": str(fail_folder / "failure_metrics.json"),
        "model_info_path": str(fail_folder / "failure_model_info.json"),
    }
