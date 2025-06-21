from mcp.server.fastmcp import FastMCP
from tools import run_full_timeseries_pipeline

mcp = FastMCP("bms_timeseries_prediction_mcp")

@mcp.tool()
def predict_cell_timeseries(
    data_path: str, cell_id: str, steps: int = 10
) -> dict:
    """
    Predict the next N steps for key battery parameters (voltage, current, temperature, etc.) for a given cell.
    Returns metrics, predictions, and plot path.
    """
    try:
        result = run_full_timeseries_pipeline(data_path, cell_id, steps)
        return result
    except Exception as e:
        return {"error": f"{str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")