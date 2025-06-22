from mcp.server.fastmcp import FastMCP
from tools import (
    run_hybrid_anomaly_detection,
    run_3d_visualization,
)

mcp = FastMCP("anomaly_detection_mcp")

@mcp.tool()
def health_check() -> dict:
    """
    Check the health of the MCP server.
    Returns a simple status message and current working directory.
    """
    import os
    return {"status": "ok", "message": "MCP server is healthy", "cwd": os.getcwd()}

@mcp.tool()
def detect_anomalies(data_path: str, chunksize: int = 100000) -> dict:
    """
    Perform hybrid anomaly detection on battery data using SVM, Isolation Forest, and VAE.
    Returns the path to the result CSV file with anomaly labels and saved models.
    """
    try:
        result_path, models = run_hybrid_anomaly_detection(data_path, chunksize)
        return {
            "status": "success",
            "message": f"Anomaly detection complete. Results saved to: {result_path}",
            "data": {"result_csv": result_path, "models": models},
            "log_path": "logs/vae/"  # Adjust if needed
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": None
        }

@mcp.tool()
def visualize_anomalies_3d(
    result_csv: str,
    anomalies_column: str = "Hybrid Anomalies",
    x_col: str = "CellVoltage",
    y_col: str = "InstantaneousCurrent",
    z_col: str = "CellTemperature",
    out_path: str = "anomaly_3d_plot.png"
) -> dict:
    """
    Generate and save a 3D scatter plot visualizing anomalies versus normal data.
    Uses specified columns for x, y, z axes and saves the plot as a PNG file.
    """
    try:
        img_path = run_3d_visualization(
            result_csv,
            anomalies_column,
            x_col,
            y_col,
            z_col,
            out_path
        )
        return {
            "status": "success",
            "message": f"3D anomaly visualization saved to: {img_path}",
            "data": {"plot": img_path},
            "log_path": None
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