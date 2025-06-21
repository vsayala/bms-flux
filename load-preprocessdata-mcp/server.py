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
import os
import logging

mcp = FastMCP("battery_preprocessing_mcp")

@mcp.tool()
def health_check() -> dict:
    return {"status": "ok", "message": "MCP server is healthy", "cwd": os.getcwd()}

def export_schema(df, out_path):
    schema = {"columns": list(df.columns), "dtypes": {col: str(df[col].dtype) for col in df.columns}}
    import json
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)
    logging.info(f"Exported schema to {out_path}")
    return out_path

@mcp.tool()
def clean_and_filter_data(data_path: str, chunksize: int = 100000) -> dict:
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = read_and_filter_chunks(data_path, chunksize)
        out_path = save_processed_data(df, run_folder, "battery_cleaned.csv")
        schema_path = export_schema(df, os.path.join(run_folder, "battery_cleaned_schema.json"))
        return {
            "status": "success",
            "message": f"Cleaned data saved to {out_path}",
            "data": {"csv": out_path, "schema": schema_path},
            "log_path": os.path.join(run_folder, "preprocess-log.log")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

@mcp.tool()
def engineer_battery_features(csv_path: str) -> dict:
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = feature_engineering(df)
        out_path = save_processed_data(df, run_folder, "battery_with_features.csv")
        schema_path = export_schema(df, os.path.join(run_folder, "battery_with_features_schema.json"))
        return {
            "status": "success",
            "message": f"Feature-engineered data saved to {out_path}",
            "data": {"csv": out_path, "schema": schema_path},
            "log_path": os.path.join(run_folder, "preprocess-log.log")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

@mcp.tool()
def impute_battery_data_mice(csv_path: str) -> dict:
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = mice_imputation(df)
        out_path = save_processed_data(df, run_folder, "battery_mice_imputed.csv")
        schema_path = export_schema(df, os.path.join(run_folder, "battery_mice_imputed_schema.json"))
        return {
            "status": "success",
            "message": f"MICE-imputed data saved to {out_path}",
            "data": {"csv": out_path, "schema": schema_path},
            "log_path": os.path.join(run_folder, "preprocess-log.log")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

@mcp.tool()
def impute_and_scale_battery_data(csv_path: str) -> dict:
    try:
        run_folder, _ = __import__("tools").setup_run_folder()
        df = pd.read_csv(csv_path)
        df = mean_impute_and_scale(df)
        out_path = save_processed_data(df, run_folder, "battery_final_scaled.csv")
        schema_path = export_schema(df, os.path.join(run_folder, "battery_final_scaled_schema.json"))
        return {
            "status": "success",
            "message": f"Data imputed (mean) and scaled. Saved to {out_path}",
            "data": {"csv": out_path, "schema": schema_path},
            "log_path": os.path.join(run_folder, "preprocess-log.log")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

@mcp.tool()
def preprocess_battery_data_tool(
    data_path: str,
    chunksize: int = 100000
) -> dict:
    try:
        msg = preprocess_battery_data(data_path, chunksize)
        # Find latest run folder
        from glob import glob
        runs = glob("runs/*/*/*/*/")
        runs.sort(reverse=True)
        run_folder = runs[0] if runs else None
        out_path = os.path.join(run_folder, "battery_preprocessed.csv") if run_folder else None
        log_path = os.path.join(run_folder, "preprocess-log.log") if run_folder else None
        schema_path = os.path.join(run_folder, "battery_preprocessed_schema.json") if run_folder else None
        # Export schema if preprocessed file exists
        if out_path and os.path.exists(out_path):
            df = pd.read_csv(out_path)
            export_schema(df, schema_path)
        return {
            "status": "success",
            "message": msg,
            "data": {"csv": out_path, "schema": schema_path},
            "log_path": log_path
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

if __name__ == "__main__":
    mcp.run(transport="stdio")