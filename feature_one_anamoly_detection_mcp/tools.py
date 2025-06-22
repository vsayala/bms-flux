import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from utils.model_utils import save_model
import joblib
from pathlib import Path
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_METAL_LOG_LEVEL"] = "1"


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


def run_hybrid_anomaly_detection(data_path, chunksize=100000):
    """
    Full anomaly detection pipeline:
        - Loads and preprocesses data
        - Runs SVM, Isolation Forest, VAE
        - Combines scores, thresholds top 5% as anomalies
        - Outputs CSV with 'Hybrid Anomalies' column
    Returns:
        - Path to results CSV
    """
    X, data = load_and_preprocess_data(data_path, chunksize)
    # Train models
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale").fit(X)
    iforest = IsolationForest(
        contamination=0.05, random_state=42, n_estimators=500, max_samples=0.9
    ).fit(X)
    svm_scores = svm.decision_function(X)
    iforest_scores = -iforest.decision_function(X)
    vae_scores = run_variational_autoencoder(X)
    hybrid_scores = hybrid_anomaly_scoring(svm_scores, iforest_scores, vae_scores)
    threshold = np.percentile(hybrid_scores, 95)
    hybrid_anomalies = (hybrid_scores > threshold).astype(int)
    data["Hybrid Anomalies"] = hybrid_anomalies
    out_path = os.path.join("data", "output", "hybrid_results.csv")
    data.to_csv(out_path, index=False)
    # Model versioning
    models_dir = "models"
    svm_path, _ = save_model(svm, "anomaly_svm", models_dir)
    iforest_path, _ = save_model(iforest, "anomaly_iforest", models_dir)
    # If using Keras, save as H5; for now, skip vae_path
    return out_path, [svm_path, iforest_path]


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


def run_anomaly_detection(data_path: str, run_folder: Path) -> dict:
    """
    Runs a hybrid Isolation Forest and Autoencoder anomaly detection model.
    Saves the model, scaler, and results to a dedicated 'anomaly_detection' folder.
    """
    anomaly_folder = run_folder / "anomaly_detection"
    anomaly_folder.mkdir(exist_ok=True)
    logging.info(
        f"Starting anomaly detection. Outputs will be saved to: {anomaly_folder}"
    )

    df = pd.read_csv(data_path)
    features = df.select_dtypes(include=["float64", "int64"])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    joblib.dump(scaler, anomaly_folder / "scaler.gz")

    # Autoencoder
    input_dim = scaled_features.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(int(input_dim / 2), activation="relu")(input_layer)
    encoder = Dense(int(input_dim / 4), activation="relu")(encoder)
    decoder = Dense(int(input_dim / 2), activation="relu")(encoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        scaled_features,
        scaled_features,
        epochs=10,
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        verbose=0,
    )
    autoencoder.save(anomaly_folder / "autoencoder_model.h5")

    # Isolation Forest
    iso_forest = IsolationForest(contamination="auto", random_state=42)
    iso_forest.fit(scaled_features)
    joblib.dump(iso_forest, anomaly_folder / "iso_forest_model.gz")

    df["iso_forest_anomaly"] = iso_forest.predict(scaled_features)

    predictions = autoencoder.predict(scaled_features)
    mse = ((scaled_features - predictions) ** 2).mean(axis=1)

    # Convert mse to a Pandas Series to use the .quantile() method
    mse_series = pd.Series(mse)
    df["autoencoder_anomaly"] = (mse_series > mse_series.quantile(0.95)).astype(int)

    results_path = anomaly_folder / "anomaly_results.csv"
    df.to_csv(results_path, index=False)

    logging.info(f"Anomaly detection complete. Results saved to {results_path}")
    return {"anomaly_folder": str(anomaly_folder), "results_path": str(results_path)}
