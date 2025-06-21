from mcp.server.fastmcp import FastMCP
from tools import (
    run_hybrid_anomaly_detection,
    run_3d_visualization,
)

mcp = FastMCP("anomaly_detection_mcp")

@mcp.tool()
def detect_anomalies(data_path: str, chunksize: int = 100000) -> str:
    """
    Perform hybrid anomaly detection on battery data using One-Class SVM, Isolation Forest, and Variational Autoencoder.
    Returns the path to the result CSV file with anomaly labels.
    """
    try:
        result_path = run_hybrid_anomaly_detection(data_path, chunksize)
        return f"Anomaly detection complete. Results saved to: {result_path}"
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def visualize_anomalies_3d(
    result_csv: str,
    anomalies_column: str = "Hybrid Anomalies",
    x_col: str = "CellVoltage",
    y_col: str = "InstantaneousCurrent",
    z_col: str = "CellTemperature",
    out_path: str = "anomaly_3d_plot.png"
) -> str:
    """
    Generate and save a 3D scatter plot visualizing anomalies versus normal data.
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
        return f"3D anomaly visualization saved to: {img_path}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")