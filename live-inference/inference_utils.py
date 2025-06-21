import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def preprocess_live_data(input_data):
    # Ensure incoming data matches training schema
    # Placeholder: adapt to your feature_engineering as in preprocessing
    df = pd.DataFrame([input_data]) if isinstance(input_data, dict) else pd.DataFrame(input_data)
    # ... (add feature engineering, imputation, scaling as in train pipeline)
    return df

def run_inference(model, df):
    preds = model.predict(df)
    return preds.tolist()