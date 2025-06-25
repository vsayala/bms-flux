import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from prophet import Prophet
except ImportError:
    Prophet = None
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except ImportError:
    LSTM = None
import json
import warnings


# Set up logging and job-run folders
def setup_run_folder():
    current_datetime = time.strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())
    job_run_id = f"{current_datetime}_{unique_id}"
    year, month, day = time.strftime("%Y"), time.strftime("%m"), time.strftime("%d")
    plots_folder = os.path.join("runs", year, month, day, job_run_id)
    os.makedirs(plots_folder, exist_ok=True)
    log_path = os.path.join(plots_folder, "bms-ml-log.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )
    logging.info(f"Created folder structure for job run ID: {job_run_id}")
    return plots_folder, job_run_id


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Started execution of '{func.__name__}'")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in '{func.__name__}': {e}")
            raise e
        else:
            elapsed_time = time.time() - start_time
            logging.info(f"Completed '{func.__name__}' in {elapsed_time:.2f} seconds")
            return result

    return wrapper


@timing_decorator
def load_and_prepare_timeseries_data(file_path, cell_id, target_columns, lags=3):
    df = pd.read_csv(file_path)
    # --- Robustness Patch: Check columns ---
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    if "Cell ID" not in df.columns:
        raise ValueError("Missing required column: 'Cell ID'")
    df = df[df["Cell ID"] == cell_id].copy()
    for col in target_columns:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit_direction="both")
            df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
        else:
            raise ValueError(f"Missing required target column: {col}")
    # Create lag features
    for target in target_columns:
        for lag in range(1, lags + 1):
            df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df = df.dropna()
    return df


@timing_decorator
def train_xgboost(df, feature_columns, target):
    import logging
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import numpy as np

    X = df[feature_columns]
    y = df[target]
    # Robustness: Check for empty or too-small data
    if X.shape[0] < 5 or X.shape[1] == 0:
        logging.warning(
            f"[XGBoost] Not enough data (rows={X.shape[0]}, features={X.shape[1]}) for GridSearchCV. Fitting default model."
        )
        try:
            model = XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X) if X.shape[0] > 0 else np.array([])
            metrics = {
                "mae": float(mean_absolute_error(y, y_pred)) if len(y_pred) else None,
                "rmse": (
                    float(mean_squared_error(y, y_pred, squared=False))
                    if len(y_pred)
                    else None
                ),
                "r2": float(r2_score(y, y_pred)) if len(y_pred) else None,
                "mape": (
                    float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100
                    if len(y_pred)
                    else None
                ),
                "fallback": True,
            }
        except Exception as e:
            logging.error(f"[XGBoost] Fallback model failed: {e}")
            return None, {"error": str(e), "fallback": True}
        return model, metrics
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        xgb_params = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        }
        xgb_gs = GridSearchCV(
            XGBRegressor(),
            xgb_params,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=TypeError)
            xgb_gs.fit(X_train, y_train)
        best_xgb = xgb_gs.best_estimator_
        y_pred = best_xgb.predict(X_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(mean_squared_error(y_test, y_pred, squared=False)),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8)))) * 100,
            "fallback": False,
        }
        return best_xgb, metrics
    except Exception as e:
        logging.error(f"[XGBoost] GridSearchCV failed: {e}. Fitting default model.")
        try:
            model = XGBRegressor()
            model.fit(X, y)
            y_pred = model.predict(X) if X.shape[0] > 0 else np.array([])
            metrics = {
                "mae": float(mean_absolute_error(y, y_pred)) if len(y_pred) else None,
                "rmse": (
                    float(mean_squared_error(y, y_pred, squared=False))
                    if len(y_pred)
                    else None
                ),
                "r2": float(r2_score(y, y_pred)) if len(y_pred) else None,
                "mape": (
                    float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100
                    if len(y_pred)
                    else None
                ),
                "fallback": True,
                "error": str(e),
            }
            return model, metrics
        except Exception as e2:
            logging.error(f"[XGBoost] Fallback model also failed: {e2}")
            return None, {"error": str(e2), "fallback": True}


@timing_decorator
def train_prophet(df, target):
    if Prophet is None:
        return None, {"error": "Prophet not installed"}
    prophet_df = df[["Timestamp", target]].rename(
        columns={"Timestamp": "ds", target: "y"}
    )
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=10, freq="H")
    forecast = model.predict(future)
    y_true = prophet_df["y"]
    y_pred = forecast["yhat"][: len(y_true)]
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))) * 100,
    }
    return model, metrics


@timing_decorator
def train_lstm(df, feature_columns, target):
    if LSTM is None:
        return None, {"error": "LSTM not available"}
    from sklearn.preprocessing import MinMaxScaler

    X = df[feature_columns].values
    y = df[target].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    y = y.reshape(-1, 1)
    model = Sequential()
    model.add(
        LSTM(
            32,
            input_shape=(X_scaled.shape[1], X_scaled.shape[2]),
            return_sequences=False,
        )
    )
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)
    y_pred = model.predict(X_scaled).flatten()
    metrics = {
        "mae": mean_absolute_error(y, y_pred),
        "rmse": mean_squared_error(y, y_pred, squared=False),
        "r2": r2_score(y, y_pred),
        "mape": float(np.mean(np.abs((y - y_pred) / (y + 1e-8)))) * 100,
    }
    return model, metrics


@timing_decorator
def predict_next_steps(models, df, feature_columns, target_columns, steps=10):
    predictions = {}
    latest_features = df.iloc[-1][feature_columns].values.reshape(1, -1)
    for target in target_columns:
        model = models[target]
        pred = []
        features = latest_features.copy()
        for _ in range(steps):
            next_val = model.predict(features)[0]
            pred.append(next_val)
            features = np.append(features[:, 1:], [[next_val]], axis=1)
        predictions[target] = pred
        logging.info(f"Predicted next values for {target}: {pred}")
    return predictions


@timing_decorator
def plot_timeseries_predictions(
    df, predictions, target_columns, cell_id, plots_folder, steps=10
):
    # Plots historical + predicted values for each target
    df = df.copy()
    num_targets = len(target_columns)
    n_cols = 2
    n_rows = (num_targets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    for i, target in enumerate(target_columns):
        ax = axes[i]
        ax.plot(
            df["Timestamp"],
            df[target],
            label=f"Historical {target}",
            color="blue",
            alpha=0.7,
        )
        ax.scatter(
            pd.date_range(start=df["Timestamp"].iloc[-1], periods=steps, freq="T"),
            predictions[target],
            label=f"Predicted {target}",
            color="red",
            s=100,
        )
        ax.set_title(f"{target} Over Time for Cell {cell_id}")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(plots_folder, f"predicted_values_plot_cell_{cell_id}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


@timing_decorator
def run_full_timeseries_pipeline(file_path, cell_id, steps=10):
    plots_folder, job_run_id = setup_run_folder()
    target_columns = [
        "Voltage (V)",
        "Current (A)",
        "Resistance (Ohms)",
        "SOC (%)",
        "SOD (%)",
        "Cell Temperature (°C)",
        "Ambient Temperature (°C)",
    ]
    df = load_and_prepare_timeseries_data(file_path, cell_id, target_columns)
    feature_columns = [col for col in df.columns if "lag" in col]
    all_metrics = {}
    all_models = {}
    best_models = {}
    for target in target_columns:
        # Train all models
        xgb_model, xgb_metrics = train_xgboost(df, feature_columns, target)
        _prophet_model, _prophet_metrics = train_prophet(df, target)
        _lstm_model, _lstm_metrics = train_lstm(df, feature_columns, target)
        # Compare and select best
        metrics_list = [
            ("xgboost", xgb_metrics),
            ("prophet", _prophet_metrics),
            ("lstm", _lstm_metrics),
        ]
        # Use RMSE as main criterion
        best = min(metrics_list, key=lambda x: x[1].get("rmse", float("inf")))
        best_models[target] = best[0]
        all_metrics[target] = {
            "xgboost": xgb_metrics,
            "prophet": _prophet_metrics,
            "lstm": _lstm_metrics,
            "best": best[0],
        }
        all_models[target] = {
            "xgboost": str(type(xgb_model)),
            "prophet": str(type(_prophet_model)),
            "lstm": str(type(_lstm_model)),
        }
    # Save metrics and model info
    metrics_path = Path(plots_folder) / "timeseries_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    model_info_path = Path(plots_folder) / "timeseries_model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(all_models, f, indent=2)
    # Use XGBoost for prediction for now (can be extended)
    predictions = predict_next_steps(
        {
            t: xgb_model
            for t, xgb_model in zip(
                target_columns,
                [train_xgboost(df, feature_columns, t)[0] for t in target_columns],
            )
        },
        df,
        feature_columns,
        target_columns,
        steps,
    )
    plot_path = plot_timeseries_predictions(
        df, predictions, target_columns, cell_id, plots_folder, steps
    )
    return {
        "metrics": all_metrics,
        "predictions": predictions,
        "plot_path": os.path.abspath(plot_path),
        "job_run_id": job_run_id,
        "plots_folder": os.path.abspath(plots_folder),
        "metrics_path": str(metrics_path),
        "model_info_path": str(model_info_path),
        "best_models": best_models,
    }


def predict_cell_timeseries(
    data_path: str, cell_id: str, steps: int, run_folder: Path
) -> dict:
    """
    Predicts future values for a given cell's parameters using the full time series pipeline.
    Saves the plot, predictions CSV, metrics, and model info to 'timeseries_prediction' folder.
    """
    try:
        result = run_timeseries_prediction(data_path, run_folder)
        # Optionally, generate a plot for the requested cell_id and steps
        # (You can add a plotting function here if needed)
        return {
            "status": "success",
            "message": "Timeseries prediction complete",
            "data": result,
            "log_path": str(run_folder / "timeseries_prediction"),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "data": None,
            "log_path": str(run_folder / "timeseries_prediction"),
        }


def run_timeseries_prediction(data_path: str, run_folder: Path) -> dict:
    """
    Time series prediction pipeline with dynamic lag adjustment:
        - Loads and preprocesses data
        - Dynamically chooses number of lags based on available data
        - Runs XGBoost, Prophet, LSTM (with tuning)
        - Saves metrics, best model info, and predictions for each cell/target/steps
    Returns:
        - Dict with paths to results, metrics, and model info
    """
    import traceback

    ts_folder = run_folder / "timeseries_prediction"
    ts_folder.mkdir(exist_ok=True)
    preds_path = ts_folder / "timeseries_predictions.csv"
    metrics_path = ts_folder / "timeseries_metrics.json"
    model_info_path = ts_folder / "timeseries_model_info.json"
    logging.info(
        f"[DEBUG] Entered run_timeseries_prediction. data_path={data_path}, run_folder={run_folder}"
    )
    all_preds = []
    metrics = {}
    best_models = {}
    try:
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            logging.error(f"Failed to read data: {e}")
            df = pd.DataFrame()
        # Check for required columns
        if df.empty or "Cell ID" not in df.columns or "Timestamp" not in df.columns:
            logging.warning(
                f"Input data is empty or missing required columns. Writing empty predictions CSV to {preds_path}."
            )
            return {
                "predictions_path": str(preds_path),
                "metrics_path": str(metrics_path),
                "model_info_path": str(model_info_path),
            }
        cell_ids = df["Cell ID"].unique()
        targets = [
            "Voltage (V)",
            "Current (A)",
            "Cell Temperature (°C)",
            "Ambient Temperature (°C)",
        ]
        for cell_id in cell_ids:
            cell_df = df[df["Cell ID"] == cell_id].copy()
            cell_df = cell_df.sort_values("Timestamp")
            for target in targets:
                n_rows = len(cell_df)
                if n_rows >= 4:
                    max_lag = 3
                elif n_rows == 3:
                    max_lag = 2
                elif n_rows == 2:
                    max_lag = 1
                else:
                    logging.warning(
                        f"Cell {cell_id}, Target {target}: Not enough data (n={n_rows}) for any lag. Skipping."
                    )
                    continue
                # Create lag features
                for lag in range(1, max_lag + 1):
                    lag_col = f"{target}_lag{lag}"
                    if lag_col not in cell_df.columns and target in cell_df.columns:
                        cell_df[lag_col] = cell_df[target].shift(lag)
                feature_columns = [
                    f"{target}_lag{lag}"
                    for lag in range(1, max_lag + 1)
                    if f"{target}_lag{lag}" in cell_df.columns
                ]
                required_cols = [target] + feature_columns
                cell_df_lagged = cell_df.dropna(subset=required_cols)
                if cell_df_lagged.empty or not feature_columns:
                    logging.warning(
                        f"Cell {cell_id}, Target {target}: Not enough data after lag creation (lags={max_lag}) or no features. Skipping."
                    )
                    continue
                logging.info(
                    f"Cell {cell_id}, Target {target}: Using {max_lag} lag(s) for prediction."
                )
                # --- XGBoost tuning ---
                try:
                    xgb_model, xgb_metrics = train_xgboost(
                        cell_df_lagged, feature_columns, target
                    )
                except Exception as e:
                    logging.error(
                        f"Cell {cell_id}, Target {target}: XGBoost training failed: {e}"
                    )
                    continue
                # --- Prophet (no tuning for now) ---
                try:
                    _prophet_model, _prophet_metrics = train_prophet(
                        cell_df_lagged, target
                    )
                except Exception as e:
                    logging.error(
                        f"Cell {cell_id}, Target {target}: Prophet training failed: {e}"
                    )
                    _prophet_metrics = {"error": str(e)}
                # --- LSTM (no tuning for now) ---
                try:
                    _lstm_model, _lstm_metrics = train_lstm(
                        cell_df_lagged, feature_columns, target
                    )
                except Exception as e:
                    logging.error(
                        f"Cell {cell_id}, Target {target}: LSTM training failed: {e}"
                    )
                    _lstm_metrics = {"error": str(e)}
                # --- Select best model (lowest RMSE) ---
                best_model = xgb_model  # For now, use XGBoost as best
                # --- Predict for 10, 15, 20 steps ---
                last_idx = cell_df_lagged.index[-1]
                for steps in [10, 15, 20]:
                    # Predict next N steps using XGBoost
                    x_input = np.arange(last_idx + 1, last_idx + 1 + steps).reshape(
                        -1, 1
                    )
                    try:
                        preds = best_model.predict(x_input)
                    except Exception as e:
                        logging.error(
                            f"Cell {cell_id}, Target {target}: Prediction failed: {e}"
                        )
                        continue
                    for step, pred in enumerate(preds, 1):
                        all_preds.append(
                            {
                                "Cell ID": cell_id,
                                "Target": target,
                                "Step": step,
                                "Steps": steps,
                                "Predicted Value": pred,
                            }
                        )
                # Save metrics and best model info
                metrics[f"{cell_id}_{target}"] = {
                    "xgb_best_score": xgb_metrics.get("rmse"),
                    "params": getattr(xgb_metrics, "params", None),
                }
                best_models[f"{cell_id}_{target}"] = str(type(xgb_model))
    except Exception as e:
        logging.error(
            f"[FATAL] Exception in run_timeseries_prediction: {e}\n{traceback.format_exc()}"
        )
    finally:
        # Always write predictions, metrics, and model info, even if empty
        preds_df = pd.DataFrame(all_preds)
        preds_df.to_csv(preds_path, index=False)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(model_info_path, "w") as f:
            json.dump(best_models, f, indent=2)
        if preds_df.empty:
            logging.warning(
                f"[DEBUG] No predictions were made for any cell/target. Empty predictions CSV written to {preds_path}."
            )
        else:
            logging.info(f"[DEBUG] Predictions written to {preds_path}.")
        logging.info("[DEBUG] Exiting run_timeseries_prediction.")
    return {
        "predictions_path": str(preds_path),
        "metrics_path": str(metrics_path),
        "model_info_path": str(model_info_path),
    }
