from fastapi import FastAPI, UploadFile, File
from eda_mcp.tools import run_eda
import shutil
import os
import time

app = FastAPI()


@app.post("/run_eda/")
def run_eda_endpoint(file: UploadFile = File(...)):
    """
    Run EDA on uploaded CSV file. Saves all outputs in a unique run folder.
    Logs all steps to both eda.log and bms_master.log.
    Returns the run folder and output files.
    """
    # Save uploaded file to a temp location
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Run EDA
    result = run_eda(temp_path)
    # Log to master log
    with open("bms_master.log", "a") as f:
        f.write(
            f"[API] {time.strftime('%Y-%m-%d %H:%M:%S')} EDA run for {file.filename} -> {result.get('run_folder')}\n"
        )
    # Clean up temp file
    os.remove(temp_path)
    return result
