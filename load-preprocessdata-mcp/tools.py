import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

def setup_run_folder():
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())
    job_run_id = f"{current_datetime}_{unique_id}"
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    run_folder = os.path.join("runs", year, month, day, job_run_id)
    os.makedirs(run_folder, exist_ok=True)
    log_path = os.path.join(run_folder, "preprocess-log.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode='a')
    logging.info(f"Created folder structure for job run ID: {job_run_id}")
    return run_folder, job_run_id

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
def read_and_filter_chunks(data_path, chunksize=100000):
    """Load data, remove outliers and sensor errors, return cleaned dataframe."""
    required_columns = [
        "CellVoltage", "CellTemperature", "InstantaneousCurrent", "AmbientTemperature", "CellSpecificGravity"
    ]
    chunk_list = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        missing_columns = [col for col in required_columns if col not in chunk.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        chunk["CellVoltage"] = chunk["CellVoltage"].apply(lambda x: np.nan if x >= 60 else x)
        chunk["CellTemperature"] = chunk["CellTemperature"].apply(lambda x: np.nan if x >= 1000 else x)
        chunk["CellSpecificGravity"] = chunk["CellSpecificGravity"].apply(lambda x: np.nan if x >= 50 else x)
        chunk = chunk.dropna(subset=["InstantaneousCurrent", "AmbientTemperature"])
        chunk_list.append(chunk)
    df = pd.concat(chunk_list, ignore_index=True)
    logging.info(f"Read and cleaned data shape: {df.shape}")
    return df

@timing_decorator
def feature_engineering(df):
    """Add engineered, lagged, rolling, and polynomial features to dataframe."""
    df["Power (W)"] = df["CellVoltage"] * df["InstantaneousCurrent"]
    df["Resistance (Ohms)"] = df["CellVoltage"] / (df["InstantaneousCurrent"] + 1e-6)
    df["Temperature Deviation"] = abs(df["CellTemperature"] - df["AmbientTemperature"])
    df["dTemperature/dt"] = df["CellTemperature"].diff().fillna(0)
    df["dVoltage/dt"] = df["CellVoltage"].diff().fillna(0)
    df["Rolling_Mean_Temperature"] = df["CellTemperature"].rolling(window=5).mean().fillna(0)
    df["Rolling_Std_Temperature"] = df["CellTemperature"].rolling(window=5).std().fillna(0)
    df["Voltage*Current"] = df["CellVoltage"] * df["InstantaneousCurrent"]
    df["Voltage^2"] = df["CellVoltage"] ** 2
    df["Lag_Voltage"] = df["CellVoltage"].shift(1).fillna(0)
    df["Lag_Current"] = df["InstantaneousCurrent"].shift(1).fillna(0)
    # Polynomial features for core numeric columns
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[["CellVoltage", "InstantaneousCurrent", "CellTemperature"]])
    poly_feature_names = poly.get_feature_names_out(["CellVoltage", "InstantaneousCurrent", "CellTemperature"])
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    df = pd.concat([df.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1)
    logging.info(f"Feature engineering complete. Data shape: {df.shape}")
    return df

@timing_decorator
def mice_imputation(df):
    """Perform robust MICE imputation on required columns."""
    required_columns = [
        "CellVoltage", "CellTemperature", "InstantaneousCurrent", "AmbientTemperature", "CellSpecificGravity"
    ]
    mice_imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = mice_imputer.fit_transform(df[required_columns])
    imputed_df = pd.DataFrame(imputed_data, columns=required_columns)
    for col in required_columns:
        df[col] = imputed_df[col]
    logging.info("MICE imputation complete.")
    return df

@timing_decorator
def mean_impute_and_scale(df):
    """Mean-impute remaining missing values and scale numeric features."""
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logging.info("Mean imputation and robust scaling complete.")
    return df

@timing_decorator
def save_processed_data(df, run_folder, file_name):
    out_path = os.path.join(run_folder, file_name)
    df.to_csv(out_path, index=False)
    logging.info(f"Saved processed data to {out_path}")
    return out_path

@timing_decorator
def preprocess_battery_data(data_path, chunksize=100000):
    run_folder, job_run_id = setup_run_folder()
    df = read_and_filter_chunks(data_path, chunksize)
    df = feature_engineering(df)
    df = mice_imputation(df)
    df = mean_impute_and_scale(df)
    out_path = save_processed_data(df, run_folder, "battery_preprocessed.csv")
    return f"Preprocessing complete. Processed data saved to: {out_path}. Logs in {run_folder}/preprocess-log.log"