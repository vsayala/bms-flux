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

    Returns processed file path (inside a unique run folder).
    """
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = read_and_filter_chunks(data_path, chunksize)
        out_path = save_processed_data(df, run_folder, "battery_cleaned.csv")
        return f"Cleaned data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def engineer_battery_features(csv_path: str) -> str:
    """
    Add engineered, lagged, rolling, and polynomial features to cleaned battery data.
    """
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = feature_engineering(df)
        out_path = save_processed_data(df, run_folder, "battery_with_features.csv")
        return f"Feature-engineered data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def impute_battery_data_mice(csv_path: str) -> str:
    """
    Perform robust multivariate MICE imputation on core sensor columns of battery data.
    """
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = mice_imputation(df)
        out_path = save_processed_data(df, run_folder, "battery_mice_imputed.csv")
        return f"MICE-imputed data saved to {out_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def impute_and_scale_battery_data(csv_path: str) -> str:
    """
    Impute remaining missing values by mean imputation and scale all numeric features for ML-readiness.
    """
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = mean_impute_and_scale(df)
        out_path = save_processed_data(df, run_folder, "battery_final_scaled.csv")
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
    """
    return preprocess_battery_data(data_path, chunksize)

if __name__ == "__main__":
    mcp.run(transport="stdio")