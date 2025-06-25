import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from pydantic import ValidationError
import logging
from schema.bms_schema import BMSSchema

logger = logging.getLogger(__name__)

# Define domain constraints for key columns
DOMAIN_LIMITS = {
    "Voltage (V)": (0, 5),
    "Cell Temperature (°C)": (0, 100),
    "SOC (%)": (0, 100),
    "SOD (%)": (0, 100),
    "ProblemCells": (0, 100),
    "CellsConnectedCount": (1, 100),
    "Cell ID": (1, 1000),
    "PacketID": (1, 1e12),
    "DeviceID": (1, 1e12),
    "BMSManufacturerID": (1, 1e12),
}

# Columns that must be integers
INT_COLUMNS = [
    "PacketID",
    "CellsConnectedCount",
    "ProblemCells",
    "Cell ID",
    "DeviceID",
    "BMSManufacturerID",
]

# Columns that must be booleans
BOOL_COLUMNS = [
    "BMSBankDischargeCycle",
    "BMSAmbientTemperatureHN",
    "BMSSocLN",
    "BMSStringCurrentHN",
    "BMSBmsSedCommunication",
    "BMSCellCommunication",
    "BMSCellVoltageLN",
    "BMSCellVoltageNH",
    "BMSCellTemperatureHN",
    "BMSBuzzer",
    "Energy",
    "ChargerInputMains",
    "ChargerACVoltageULN",
    "ChargerLoad",
    "ChargerTrip",
    "ChargerOutputMccb",
    "ChargerBatteryCondition",
    "ChargerTestPushButton",
    "ChargerResetPushButton",
    "ChargerAlarmSupplyFuse",
    "ChargerFilterFuse",
    "ChargerOutputFuse",
    "ChargerInputFuse",
]

# Columns to exclude from imputation (IDs, timestamps, categoricals)
EXCLUDE_IMPUTE = [
    "PacketID",
    "StartPacket",
    "DataIdentifier",
    "SiteID",
    "Time",
    "Date",
    "Timestamp",
    "DeviceID",
    "BMSManufacturerID",
    "SerialNumber",
    "InstallationDate",
    "Cell ID",
    "CellServerTime",
    "ChargerRectifierFuse",
]

# Categorical columns to impute with mode
CATEGORICAL_COLUMNS = [
    "StartPacket",
    "DataIdentifier",
    "SiteID",
    "SerialNumber",
    "InstallationDate",
    "ChargerRectifierFuse",
]


def validate_data(df: pd.DataFrame, num_rows_to_check: int = 100):
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
    Main function to preprocess the battery data using MICE and domain-aware corrections.
    Includes advanced imputation for categorical columns and custom logic for specific fields.
    Adds custom rules for error codes and edge cases.
    """
    raw_df = pd.read_csv(data_path)
    validate_data(raw_df)
    df = raw_df.copy()

    # --- Robust renaming for required columns ---
    rename_map = {
        "CellVoltage": "Voltage (V)",
        "InstantaneousCurrent": "Current (A)",
        "CellTemperature": "Cell Temperature (°C)",
        "CellNumber": "Cell ID",
        "SocLatestValueForEveryCycle": "SOC (%)",
        "DodLatestValueForEveryCycle": "SOD (%)",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # --- Custom error code and edge case handling ---
    df["is_edge_case"] = False
    if "ErrorCode" in df.columns:
        # Example: If ErrorCode is not 0 or NaN, flag as edge case
        df["is_edge_case"] = df["ErrorCode"].notnull() & (df["ErrorCode"] != 0)
        # Optionally, handle specific codes (e.g., 9999 = sensor failure)
        if (df["ErrorCode"] == 9999).any():
            logger.warning(
                "Rows with ErrorCode 9999 (sensor failure) found. Setting all values to NaN for these rows."
            )
            df.loc[df["ErrorCode"] == 9999, :] = np.nan
            df.loc[df["ErrorCode"] == 9999, "is_edge_case"] = True

    # --- Outlier/rare value handling ---
    # Example: Negative voltages, temperatures, or impossible combinations
    for col, (low, high) in DOMAIN_LIMITS.items():
        if col in df.columns:
            outlier_mask = (df[col] < low) | (df[col] > high)
            if outlier_mask.any():
                logger.warning(
                    f"Edge case: {outlier_mask.sum()} outlier(s) in {col} (outside [{low}, {high}]) flagged."
                )
                df.loc[outlier_mask, "is_edge_case"] = True

    # 1. Impute categorical columns with mode
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "UNKNOWN"
            df[col] = df[col].fillna(fill_value)

    # 2. Impute booleans with mode
    for col in BOOL_COLUMNS:
        if col in df.columns:
            mode = df[col].mode(dropna=True)
            fill_value = bool(mode.iloc[0]) if not mode.empty else False
            df[col] = df[col].fillna(fill_value)

    # 3. Custom logic for timestamps: reconstruct if missing
    if "Timestamp" not in df.columns or df["Timestamp"].isnull().all():
        if "Date" in df.columns and "Time" in df.columns:
            df["Timestamp"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], errors="coerce"
            )
        else:
            df["Timestamp"] = pd.Timestamp("1970-01-01")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # 4. Custom logic for IDs/serials: fill missing with placeholder
    for col in ["PacketID", "DeviceID", "BMSManufacturerID", "Cell ID"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
    for col in ["SerialNumber", "InstallationDate"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN")

    # 5. Custom logic for counts: set negatives to 0
    for col in ["ProblemCells", "CellsConnectedCount"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Ensure 'Cell ID' is present and correct
    if "Cell ID" not in df.columns:
        if "CellNumber" in df.columns:
            df["Cell ID"] = df["CellNumber"]
        else:
            df["Cell ID"] = 1  # Placeholder if all else fails
    df["Cell ID"] = df["Cell ID"].fillna(1).astype(int)

    # 6. Numeric imputation (MICE) for continuous columns
    numeric_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col not in EXCLUDE_IMPUTE
    ]
    # Only impute columns with at least one non-missing value
    cols_with_data = [col for col in numeric_cols if not df[col].isnull().all()]
    cols_all_nan = [col for col in numeric_cols if df[col].isnull().all()]
    if cols_with_data:
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=42,
            max_iter=20,
            sample_posterior=True,
        )
        imputed = imputer.fit_transform(df[cols_with_data])
        df[cols_with_data] = imputed
    if cols_all_nan:
        logger.warning(f"Dropping columns with all NaN values: {cols_all_nan}")
        df.drop(columns=cols_all_nan, inplace=True)

    # 7. Domain-aware postprocessing
    for col, (low, high) in DOMAIN_LIMITS.items():
        if col in df.columns:
            df[col] = np.clip(df[col], low, high)
    for col in INT_COLUMNS:
        if col in df.columns:
            df[col] = np.round(df[col]).astype(int)
    for col in BOOL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: bool(int(x)) if not pd.isnull(x) else False
            )

    # 8. Final validation: check for any remaining invalid values
    for col, (low, high) in DOMAIN_LIMITS.items():
        if col in df.columns:
            if (df[col] < low).any() or (df[col] > high).any():
                logger.warning(
                    f"Column {col} has values outside domain limits after imputation."
                )

    # 9. Save preprocessed data
    out_path = Path("data/output/preprocessed_battery_data.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Preprocessed data saved to {out_path}")
    return str(out_path)


# Note: The other functions from the original file (feature_engineering, impute_missing_values, etc.)
# are intentionally removed as their logic is now consolidated into the main preprocess_battery_data function.
