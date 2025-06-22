import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import os
import time
from pydantic import ValidationError
import logging

from schema.bms_schema import BMSSchema

logger = logging.getLogger(__name__)


def setup_run_folder():
    """Sets up a unique directory for the current run to store outputs."""
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    run_folder = os.path.join("runs", year, month, day, f"run_{current_datetime}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def validate_data(df: pd.DataFrame, num_rows_to_check: int = 100):
    """
    Validates the first N rows of a DataFrame against the BMSSchema.
    Raises a ValueError if validation fails.
    """
    logger.info(f"Validating the first {num_rows_to_check} rows of the dataset...")
    sample_records = df.head(num_rows_to_check).to_dict(orient="records")

    for i, record in enumerate(sample_records):
        try:
            BMSSchema(**record)
        except ValidationError as e:
            error_msg = f"Data validation failed on row {i+2}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    logger.info("Data validation successful.")


def preprocess_battery_data(data_path: str) -> str:
    """
    Loads, cleans, and preprocesses the battery data in a single, robust function.
    Returns the path to the preprocessed CSV file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.join("data", "output"), exist_ok=True)
    df = pd.read_csv(data_path)

    # Rename columns to match pipeline expectations
    df.rename(
        columns={
            "CellNumber": "Cell ID",
            "PacketDateTime": "Timestamp",
            "CellVoltage": "Voltage (V)",
            "InstantaneousCurrent": "Current (A)",
            "SocLatestValueForEveryCycle": "SOC (%)",
            "DodLatestValueForEveryCycle": "SOD (%)",
            "CellTemperature": "Cell Temperature (°C)",
            "AmbientTemperature": "Ambient Temperature (°C)",
        },
        inplace=True,
    )

    # Check that required columns exist, but keep all columns
    required_cols = [
        "Cell ID",
        "Timestamp",
        "Voltage (V)",
        "Cell Temperature (°C)",
        "Current (A)",
        "Ambient Temperature (°C)",
        "CellSpecificGravity",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert Timestamp to a numeric format if it exists
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if not df["Timestamp"].isnull().all():
            df["Timestamp"] = (
                df["Timestamp"] - pd.Timestamp("1970-01-01")
            ) // pd.Timedelta("1s")
        else:
            df["Timestamp"] = 0  # Fill with a placeholder if all are invalid

    # Impute missing values only in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_with_data = [col for col in numeric_cols if not df[col].isnull().all()]
    cols_all_nan = [col for col in numeric_cols if df[col].isnull().all()]

    if cols_with_data:
        imputer = SimpleImputer(strategy="mean")
        df[cols_with_data] = imputer.fit_transform(df[cols_with_data])

    # Optionally, drop columns that are all NaN
    if cols_all_nan:
        df.drop(columns=cols_all_nan, inplace=True)

    # --- FIX: Ensure 'Cell ID' is an integer after imputation ---

    # Basic feature engineering
    if "Voltage (V)" in df.columns and "Current (A)" in df.columns:
        df["Power (W)"] = df["Voltage (V)"] * df["Current (A)"]
        df["Resistance (Ohms)"] = df["Voltage (V)"] / (df["Current (A)"] + 1e-6)

    # Scale numeric features, excluding IDs and targets
    numeric_cols_to_scale = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_exclude_from_scaling = [
        "Cell ID",
        "IsDead",
        "Timestamp",
        "SOC (%)",
        "SOD (%)",
    ]
    scale_cols = [
        col for col in numeric_cols_to_scale if col not in cols_to_exclude_from_scaling
    ]

    if scale_cols:
        scaler = RobustScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # --- FIX: Ensure 'Cell ID' is an integer after all processing ---
    if "Cell ID" in df.columns:
        df["Cell ID"] = df["Cell ID"].astype(int)

    # After renaming, keep a copy for backward compatibility
    if "Voltage (V)" in df.columns:
        df["CellVoltage"] = df["Voltage (V)"]
    if "Current (A)" in df.columns:
        df["InstantaneousCurrent"] = df["Current (A)"]

    # Save the cleaned data
    output_path = os.path.join("data", "output", "battery_preprocessed.csv")
    df.to_csv(output_path, index=False)

    return output_path


# Note: The other functions from the original file (feature_engineering, impute_missing_values, etc.)
# are intentionally removed as their logic is now consolidated into the main preprocess_battery_data function.
