from mcp.server.fastmcp import FastMCP
from tools import run_full_failure_prediction_pipeline

mcp = FastMCP("bms_failure_prediction_mcp")

@mcp.tool()
def predict_cell_failure(
    data_path: str
) -> dict:
    """
    Predict cells likely to fail in the near future.
    Returns accuracy, failing cell IDs, and job run info.
    """
    try:
        result = run_full_failure_prediction_pipeline(data_path)
        return result
    except Exception as e:
        return {"error": f"{str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")