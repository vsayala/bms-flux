import os
import time
import uuid
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path


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
def train_timeseries_models(df, target_columns):
    feature_columns = [col for col in df.columns if "lag" in col]
    models = {}
    metrics = {}
    for target in target_columns:
        X = df[feature_columns]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[target] = {"mse": mse, "r2": r2}
        models[target] = model
        logging.info(f"{target} - MSE: {mse:.4f}, R2: {r2:.4f}")
    return models, metrics, feature_columns


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
    models, metrics, feature_columns = train_timeseries_models(df, target_columns)
    predictions = predict_next_steps(models, df, feature_columns, target_columns, steps)
    plot_path = plot_timeseries_predictions(
        df, predictions, target_columns, cell_id, plots_folder, steps
    )
    return {
        "metrics": metrics,
        "predictions": predictions,
        "plot_path": os.path.abspath(plot_path),
        "job_run_id": job_run_id,
        "plots_folder": os.path.abspath(plots_folder),
    }


def predict_cell_timeseries(
    data_path: str, cell_id: str, steps: int, run_folder: Path
) -> dict:
    """
    Predicts future values for a given cell's voltage using an ARIMA model.
    Saves the plot to a dedicated 'timeseries_prediction' folder.
    """
    timeseries_folder = run_folder / "timeseries_prediction"
    timeseries_folder.mkdir(exist_ok=True)
    logging.info(
        f"Starting time series prediction. Outputs will be saved to: {timeseries_folder}"
    )

    df = pd.read_csv(data_path, parse_dates=["Timestamp"])
    cell_data = df[df["Cell ID"] == int(cell_id)]

    if cell_data.empty:
        logging.error(f"No data found for Cell ID {cell_id}")
        return {"error": f"No data for Cell ID {cell_id}"}

    series = cell_data.set_index("Timestamp")["Voltage (V)"]

    # ARIMA model
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series, label="Historical")
    plt.plot(forecast.index, forecast, label="Forecast")
    plt.title(f"Voltage Forecast for Cell {cell_id}")
    plt.legend()

    plot_path = timeseries_folder / f"forecast_cell_{cell_id}.png"
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Time series forecast complete. Plot saved to {plot_path}")
    return {"timeseries_folder": str(timeseries_folder), "plot_path": str(plot_path)}
