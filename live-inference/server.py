import os
import sys
import json
import pandas as pd

# --- Robustness Patch: Import utilities regardless of run context/location ---
try:
    from utils.model_utils import load_latest_model
    from schema.bms_schema import get_bms_columns
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.model_utils import load_latest_model
    from schema.bms_schema import get_bms_columns

def predict_live(data: dict, feature: str):
    # Validate input schema
    expected_cols = get_bms_columns()
    for col in expected_cols:
        if col not in data:
            return {"status": "error", "message": f"Missing required field: {col}", "data": None}
    df = pd.DataFrame([data])
    try:
        model = load_latest_model(feature)
        y_pred = model.predict(df[expected_cols])
    except Exception as e:
        return {"status": "error", "message": f"Model inference error: {str(e)}", "data": None}
    return {"status": "success", "message": "Prediction complete", "data": {"prediction": y_pred.tolist()}}

if __name__ == "__main__":
    feature = sys.argv[1] if len(sys.argv) > 1 else "anomaly_svm"
    try:
        data = json.loads(sys.stdin.read())
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Invalid input JSON: {e}", "data": None}))
        sys.exit(1)
    result = predict_live(data, feature)
    print(json.dumps(result))