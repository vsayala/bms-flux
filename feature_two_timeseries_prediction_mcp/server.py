# Update the import path if FastMCP is in a different location, e.g.:
# from fastmcp import FastMCP
# or
# from .fastmcp import FastMCP

from mcp.server.fastmcp import FastMCP
from feature_two_timeseries_prediction_mcp.tools import run_full_timeseries_pipeline

mcp = FastMCP("bms_timeseries_prediction_mcp")


@mcp.tool()
def predict_cell_timeseries(data_path: str, cell_id: str, steps: int = 10) -> dict:
    """
    Predict the next N steps for key battery parameters (voltage, current, temperature, etc.) for a given cell.
    Returns metrics, predictions, and plot path.
    """
    try:
        result = run_full_timeseries_pipeline(data_path, cell_id, steps)
        return {
            "status": "success",
            "message": "Timeseries prediction complete",
            "data": result,
            "log_path": result.get("plots_folder", None),
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": None, "log_path": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")
