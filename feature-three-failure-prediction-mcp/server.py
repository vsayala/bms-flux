from mcp.server.fastmcp import FastMCP
from tools import run_full_failure_prediction_pipeline

mcp = FastMCP("bms_failure_prediction_mcp")

@mcp.tool()
def health_check() -> dict:
    import os
    return {"status": "ok", "message": "MCP server is healthy", "cwd": os.getcwd()}

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
        return {
            "status": "success",
            "message": "Failure prediction complete",
            "data": result,
            "log_path": result.get('plots_folder', None)
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