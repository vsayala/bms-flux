from mcp.server.fastmcp import FastMCP
from tools import (
    read_and_filter_chunks,
    feature_engineering,
    mice_imputation,
    mean_impute_and_scale,
    save_processed_data,
    preprocess_battery_data,
)
import pandas as pd

mcp = FastMCP("battery_preprocessing_mcp")

@mcp.tool()
def clean_and_filter_data(data_path: str, chunksize: int = 100000) -> str:
    """
    Load raw battery CSV data, remove outliers and obvious sensor errors, and return a temporary CSV path.

    This tool reads battery data in chunks, applies:
      - Outlier cleaning for voltage, temperature, and specific gravity
      - Removal of rows where critical columns (current, ambient temp) are missing

    Args:
        data_path (str): Path to the raw battery CSV file.
        chunksize (int, optional): Number of rows to load per chunk (default: 100000).

    Returns:
        str: Path to a temporary cleaned CSV file, or an error message.
    """
    try:
        df = read_and_filter_chunks(data_path, chunksize)
        out_path = "battery_cleaned.csv"
        df.to_csv(out_path, index=False)
        return f"Cleaned data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def engineer_battery_features(csv_path: str) -> str:
    """
    Add engineered, lagged, rolling, and polynomial features to cleaned battery data.

    Args:
        csv_path (str): Path to a cleaned battery CSV file.

    Returns:
        str: Path to a CSV file with added features, or an error message.
    """
    try:
        df = pd.read_csv(csv_path)
        df = feature_engineering(df)
        out_path = "battery_with_features.csv"
        df.to_csv(out_path, index=False)
        return f"Feature-engineered data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def impute_battery_data_mice(csv_path: str) -> str:
    """
    Perform robust multivariate MICE imputation on core sensor columns of battery data.

    Args:
        csv_path (str): Path to a battery CSV file with features.

    Returns:
        str: Path to a CSV with core columns imputed, or an error message.
    """
    try:
        df = pd.read_csv(csv_path)
        df = mice_imputation(df)
        out_path = "battery_mice_imputed.csv"
        df.to_csv(out_path, index=False)
        return f"MICE-imputed data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def impute_and_scale_battery_data(csv_path: str) -> str:
    """
    Impute remaining missing values by mean imputation and scale all numeric features for ML-readiness.

    Args:
        csv_path (str): Path to a MICE-imputed battery CSV file.

    Returns:
        str: Path to a CSV with scaled features (for reference), or an error message.
    """
    try:
        df = pd.read_csv(csv_path)
        _ = mean_impute_and_scale(df)
        out_path = "battery_final_scaled.csv"
        df.to_csv(out_path, index=False)
        return f"Data imputed (mean) and scaled. Saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def preprocess_battery_data_tool(
    data_path: str,
    chunksize: int = 100000
) -> str:
    """
    Run the complete battery data preprocessing pipeline in one step.

    This tool:
      - Loads and cleans outliers/signals
      - Engineers features (power, resistance, rolling, lagged, polynomial, etc.)
      - Imputes robustly (MICE then mean)
      - Scales all numeric features for ML
      - Saves a single ready-to-use CSV

    Args:
        data_path (str): Path to the raw battery CSV file.
        chunksize (int, optional): Number of rows to load at once (default: 100000).

    Returns:
        str: Success message with the path to the processed CSV file, or an error message.
    """
    return preprocess_battery_data(data_path, chunksize)

if __name__ == "__main__":
    mcp.run(transport="stdio")