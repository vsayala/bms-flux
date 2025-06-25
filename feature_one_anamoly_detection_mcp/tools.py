import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from pathlib import Path
import logging
import plotly.express as px
from sklearn.model_selection import GridSearchCV

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_METAL_LOG_LEVEL"] = "1"

logger = logging.getLogger(__name__)


# -------- Data Loading and Preparation --------
def load_and_preprocess_data(data_path, chunksize=100000):
    """
    Loads and preprocesses battery sensor data for anomaly detection.
    Handles outlier cleaning, feature engineering, MICE imputation, and scaling.
    Input CSV must contain columns:
        - Voltage (V), Cell Temperature (°C), Current (A), Ambient Temperature (°C), CellSpecificGravity
    Returns:
        - scaled_features: np.ndarray of features for modeling
        - data: pd.DataFrame (original + engineered columns)
    """
    required_columns = [
        "Voltage (V)",
        "Cell Temperature (°C)",
        "Current (A)",
        "Ambient Temperature (°C)",
        "CellSpecificGravity",
    ]
    chunk_list = []
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        missing_columns = [col for col in required_columns if col not in chunk.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        chunk["Voltage (V)"] = chunk["Voltage (V)"].apply(
            lambda x: np.nan if x >= 60 else x
        )
        chunk["Cell Temperature (°C)"] = chunk["Cell Temperature (°C)"].apply(
            lambda x: np.nan if x >= 1000 else x
        )
        chunk["CellSpecificGravity"] = chunk["CellSpecificGravity"].apply(
            lambda x: np.nan if x >= 50 else x
        )
        chunk = chunk.dropna(subset=["Current (A)", "Ambient Temperature (°C)"])
        chunk["Power (W)"] = chunk["Voltage (V)"] * chunk["Current (A)"]
        chunk["Resistance (Ohms)"] = chunk["Voltage (V)"] / (
            chunk["Current (A)"] + 1e-6
        )
        chunk["Temperature Deviation"] = abs(
            chunk["Cell Temperature (°C)"] - chunk["Ambient Temperature (°C)"]
        )
        chunk["dTemperature/dt"] = chunk["Cell Temperature (°C)"].diff().fillna(0)
        chunk["dVoltage/dt"] = chunk["Voltage (V)"].diff().fillna(0)
        chunk["Rolling_Mean_Temperature"] = (
            chunk["Cell Temperature (°C)"].rolling(window=5).mean().fillna(0)
        )
        chunk["Rolling_Std_Temperature"] = (
            chunk["Cell Temperature (°C)"].rolling(window=5).std().fillna(0)
        )
        chunk["Voltage*Current"] = chunk["Voltage (V)"] * chunk["Current (A)"]
        chunk["Voltage^2"] = chunk["Voltage (V)"] ** 2
        chunk["Lag_Voltage"] = chunk["Voltage (V)"].shift(1).fillna(0)
        chunk["Lag_Current"] = chunk["Current (A)"].shift(1).fillna(0)
        chunk_list.append(chunk)
    data = pd.concat(chunk_list, ignore_index=True)
    mice_imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = mice_imputer.fit_transform(data[required_columns])
    imputed_df = pd.DataFrame(imputed_data, columns=required_columns)
    for col in required_columns:
        data[col] = imputed_df[col]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(
        data[["Voltage (V)", "Current (A)", "Cell Temperature (°C)"]]
    )
    poly_feature_names = poly.get_feature_names_out(
        ["Voltage (V)", "Current (A)", "Cell Temperature (°C)"]
    )
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    data = pd.concat(
        [data.reset_index(drop=True), poly_df.reset_index(drop=True)], axis=1
    )
    imputer = SimpleImputer(strategy="mean")
    features = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, data


# --------- Anomaly Detection Models ----------
def run_one_class_svm(X, nu=0.05, kernel="rbf", gamma="scale"):
    """
    Trains a One-Class SVM for anomaly detection.
    Returns SVM decision function scores (lower = more anomalous).
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    return model.decision_function(X)


def run_isolation_forest(X, contamination=0.05):
    """
    Trains an Isolation Forest for anomaly detection.
    Returns Isolation Forest anomaly scores (higher = more anomalous).
    """
    model = IsolationForest(
        contamination=contamination, random_state=42, n_estimators=500, max_samples=0.9
    )
    model.fit(X)
    return -model.decision_function(X)


def run_variational_autoencoder(X):
    """
    Trains a Variational Autoencoder and uses reconstruction error as anomaly score.
    Returns array of reconstruction errors.
    """
    input_dim = X.shape[1]
    latent_dim = 8
    log_dir = "logs/vae/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    inputs = Input(shape=(input_dim,))
    h = Dense(256, activation="relu")(inputs)
    h = Dropout(0.2)(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    decoder_h = Dense(256, activation="relu")
    decoder_mean = Dense(input_dim, activation="sigmoid")
    h_decoded = decoder_h(z)
    outputs = decoder_mean(h_decoded)
    vae = Model(inputs, outputs)
    vae.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    vae.fit(
        X, X, epochs=10, batch_size=1024, verbose=1, callbacks=[tensorboard_callback]
    )
    reconstructed = vae.predict(X)
    return np.mean(np.square(X - reconstructed), axis=1)


def hybrid_anomaly_scoring(svm_scores, iforest_scores, vae_scores):
    """
    Combines scores from SVM, Isolation Forest, and VAE into a single hybrid anomaly score (normalized sum).
    """

    # Normalize and sum for hybrid score
    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    return norm(svm_scores) + norm(iforest_scores) + norm(vae_scores)


def run_hybrid_anomaly_detection(data_path: str, run_folder: Path) -> dict:
    """
    Full anomaly detection pipeline with hyperparameter tuning:
        - Loads and preprocesses data
        - Runs SVM, Isolation Forest, VAE (with tuning)
        - Combines scores, thresholds top 5% as anomalies
        - Outputs CSV with 'Hybrid Anomalies' column
        - Saves metrics and model info
    Returns:
        - Dict with paths to results, metrics, and model info
    """
    anomaly_folder = run_folder / "anomaly"
    anomaly_folder.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    features = ["Voltage (V)", "Current (A)", "Cell Temperature (°C)"]
    if not all(feature in df.columns for feature in features):
        msg = f"Required features for anomaly detection not found in data: {features}"
        logger.error(msg)
        return {"error": msg}
    X = df[features].copy()
    X.fillna(X.mean(), inplace=True)
    # Propagate is_edge_case if present
    if "is_edge_case" in df.columns:
        edge_case_col = df["is_edge_case"].copy()
    else:
        edge_case_col = pd.Series([False] * len(df))
    # --- Hyperparameter tuning for Isolation Forest ---
    if_params = {
        "n_estimators": [100, 300, 500],
        "max_samples": [0.7, 0.9, 1.0],
        "contamination": [0.03, 0.05, 0.1],
    }
    if_gs = GridSearchCV(
        IsolationForest(random_state=42),
        if_params,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    if_gs.fit(X)
    best_iforest = if_gs.best_estimator_
    # --- Hyperparameter tuning for One-Class SVM ---
    svm_params = {
        "nu": [0.03, 0.05, 0.1],
        "kernel": ["rbf"],
        "gamma": ["scale", 0.1, 1],
    }
    svm_gs = GridSearchCV(
        OneClassSVM(), svm_params, cv=3, scoring="accuracy", n_jobs=-1
    )
    svm_gs.fit(X)
    best_svm = svm_gs.best_estimator_
    # VAE (no tuning for now)
    vae_scores = run_variational_autoencoder(X)

    # Hybrid score
    def norm(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

    svm_scores = best_svm.decision_function(X)
    iforest_scores = -best_iforest.decision_function(X)
    hybrid_scores = norm(svm_scores) + norm(iforest_scores) + norm(vae_scores)
    threshold = np.percentile(hybrid_scores, 95)
    hybrid_anomalies = (hybrid_scores > threshold).astype(int)
    df["hybrid_anomaly_score"] = hybrid_scores
    df["hybrid_anomaly"] = hybrid_anomalies
    df["anomaly_label"] = df["hybrid_anomaly"].apply(
        lambda x: "Anomaly" if x == 1 else "Normal"
    )
    df["is_edge_case"] = edge_case_col
    # Save results
    results_csv_path = anomaly_folder / "anomaly_results.csv"
    df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved hybrid anomaly detection results to {results_csv_path}")
    # Save model info
    model_info = {
        "svm": str(type(best_svm)),
        "iforest": str(type(best_iforest)),
        "vae": "keras.Model",
        "params": {
            "svm": best_svm.get_params(),
            "iforest": best_iforest.get_params(),
            "svm_best_score": svm_gs.best_score_,
            "iforest_best_score": if_gs.best_score_,
            "threshold": float(threshold),
        },
        "model_files": {
            "svm": str(anomaly_folder / "anomaly_svm_model.joblib"),
            "iforest": str(anomaly_folder / "anomaly_iforest_model.joblib"),
        },
    }
    import joblib

    joblib.dump(best_svm, anomaly_folder / "anomaly_svm_model.joblib")
    joblib.dump(best_iforest, anomaly_folder / "anomaly_iforest_model.joblib")
    model_info_path = anomaly_folder / "model_info.json"
    import json

    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    # Save metrics (unsupervised: proportion flagged, score stats)
    metrics = {
        "proportion_flagged": float(np.mean(hybrid_anomalies)),
        "score_mean": float(np.mean(hybrid_scores)),
        "score_std": float(np.std(hybrid_scores)),
        "score_min": float(np.min(hybrid_scores)),
        "score_max": float(np.max(hybrid_scores)),
        "n_edge_cases": int(edge_case_col.sum()),
    }
    metrics_path = anomaly_folder / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    # 3D plot
    plot_html_path = create_and_save_3d_anomaly_plot(df, anomaly_folder)
    return {
        "anomaly_results_path": str(results_csv_path),
        "anomaly_plot_path": plot_html_path,
        "metrics_path": str(metrics_path),
        "model_info_path": str(model_info_path),
        "svm_model_file": str(anomaly_folder / "anomaly_svm_model.joblib"),
        "iforest_model_file": str(anomaly_folder / "anomaly_iforest_model.joblib"),
    }


def run_3d_visualization(
    csv_path,
    anomalies_column="Hybrid Anomalies",
    x_col="Voltage (V)",
    y_col="Current (A)",
    z_col="Cell Temperature (°C)",
    out_path="anomaly_3d_plot.png",
):
    """
    Generates a 3D scatter plot of anomalies vs. normal points using three features.
    Saves plot as PNG and returns absolute path.
    """
    data = pd.read_csv(csv_path)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    anomalies = data[data[anomalies_column] == 1]
    normal = data[data[anomalies_column] == 0]
    ax.scatter(
        normal[x_col],
        normal[y_col],
        normal[z_col],
        c="blue",
        label="Normal",
        alpha=0.6,
        edgecolor="k",
    )
    ax.scatter(
        anomalies[x_col],
        anomalies[y_col],
        anomalies[z_col],
        c="red",
        label="Anomaly",
        alpha=0.8,
        edgecolor="k",
    )
    ax.set_title("3D Anomaly Visualization: Voltage, Current, Temperature", fontsize=16)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_zlabel(z_col, fontsize=12)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    return os.path.abspath(out_path)


def create_and_save_3d_anomaly_plot(df: pd.DataFrame, output_path: Path):
    """
    Creates an interactive 3D scatter plot of anomalies and saves it as an HTML file.
    """
    logger.info("Generating 3D anomaly plot...")

    # This function now expects 'anomaly_label' to be in the DataFrame
    fig = px.scatter_3d(
        df,
        x="Voltage (V)",
        y="Current (A)",
        z="Cell Temperature (°C)",
        color="anomaly_label",
        color_discrete_map={"Anomaly": "red", "Normal": "blue"},
        hover_data=["Cell ID"],
        title="3D Anomaly Detection Scatter Plot (All Cells)",
    )

    fig.update_layout(legend_title_text="Status")
    plot_filepath = output_path / "3D_anomaly_plot.html"
    fig.write_html(plot_filepath)
    logger.info(f"Saved global 3D anomaly plot to {plot_filepath}")
    return str(plot_filepath)


def run_anomaly_detection(data_path: str, run_folder: Path) -> dict:
    """
    Runs anomaly detection on the preprocessed data using Isolation Forest.
    """
    # 1. Setup anomaly-specific output folder and logger
    anomaly_folder = run_folder / "anomaly"
    anomaly_folder.mkdir(exist_ok=True)
    # ... (logger setup remains the same) ...

    # 2. Load data
    df = pd.read_csv(data_path)
    logger.info("Anomaly detection data loaded successfully.")

    # 3. Select features and run model
    # CORRECTED: Use the renamed columns from the preprocessing step
    features = ["Voltage (V)", "Current (A)", "Cell Temperature (°C)"]

    if not all(feature in df.columns for feature in features):
        msg = f"Required features for anomaly detection not found in data: {features}"
        logger.error(msg)
        return {"error": msg}

    X = df[features].copy()
    X.fillna(X.mean(), inplace=True)

    model = IsolationForest(contamination="auto", random_state=42)
    preds = model.fit_predict(X)
    scores = model.decision_function(X)

    # 4. Add results to DataFrame
    df["anomaly_score"] = scores
    df["anomaly"] = preds  # -1 for anomalies, 1 for normal.

    # Create the label column needed for plotting BEFORE saving
    df["anomaly_label"] = df["anomaly"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    # 5. Save results CSV
    results_csv_path = anomaly_folder / "anomaly_results.csv"
    df.to_csv(results_csv_path, index=False)
    logger.info(f"Saved anomaly detection results to {results_csv_path}")

    # 6. Create and save the 3D plot
    plot_html_path = create_and_save_3d_anomaly_plot(df, anomaly_folder)

    return {
        "anomaly_results_path": str(results_csv_path),
        "anomaly_plot_path": plot_html_path,
    }
