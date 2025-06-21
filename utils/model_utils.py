import os
import pickle
from datetime import datetime

def save_model(model, feature: str, models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"{feature}_model_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    latest_symlink = os.path.join(models_dir, f"{feature}_model_latest.pkl")
    try:
        if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
            os.remove(latest_symlink)
        os.symlink(model_path, latest_symlink)
    except Exception:
        # On Windows, symlink may not be allowed, so copy instead
        import shutil
        shutil.copy2(model_path, latest_symlink)
    return model_path, latest_symlink

def load_latest_model(feature, models_dir="models"):
    symlink = os.path.join(models_dir, f"{feature}_model_latest.pkl")
    if not os.path.exists(symlink):
        raise FileNotFoundError(f"Latest model symlink not found for {feature}")
    with open(symlink, "rb") as f:
        model = pickle.load(f)
    return model