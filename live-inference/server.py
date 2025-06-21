import os
import pandas as pd
from utils.model_utils import load_latest_model
from schema.bms_schema import get_bms_columns

def predict_live(data: dict, feature: str):
    # Validate input schema
    expected_cols = get_bms_columns()
    for col in expected_cols:
        if col not in data:
            return {"status": "error", "message": f"Missing required field: {col}", "data": None}
    df = pd.DataFrame([data])
    model = load_latest_model(feature)
    y_pred = model.predict(df[expected_cols])
    return {"status": "success", "message": "Prediction complete", "data": {"prediction": y_pred.tolist()}}

if __name__ == "__main__":
    import sys
    import json
    feature = sys.argv[1] if len(sys.argv) > 1 else "anomaly_svm"
    data = json.loads(sys.stdin.read())
    result = predict_live(data, feature)
    print(result)