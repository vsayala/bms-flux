import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import json
import zipfile
import os
import subprocess
from sklearn.metrics import roc_curve, auc

# --- Metric/model explanations (move to top) ---
METRIC_EXPLAIN = {
    "f1": "F1 Score: Harmonic mean of precision and recall.",
    "accuracy": "Accuracy: Proportion of correct predictions.",
    "precision": "Precision: True positives / (True positives + False positives).",
    "recall": "Recall: True positives / (True positives + False negatives).",
    "proportion_flagged": "Proportion of data flagged as anomaly.",
    "score_mean": "Mean of anomaly scores.",
    "score_std": "Standard deviation of anomaly scores.",
    "score_min": "Minimum anomaly score.",
    "score_max": "Maximum anomaly score.",
    "n_edge_cases": "Number of edge cases flagged during preprocessing.",
    "best_model": "The model with the best performance metric (F1, RMSE, etc.).",
    "threshold": "Threshold used to flag anomalies.",
}
MODEL_EXPLAIN = {
    "XGBClassifier": "Extreme Gradient Boosting Classifier (XGBoost)",
    "RandomForestClassifier": "Random Forest Classifier",
    "LGBMClassifier": "LightGBM Classifier",
    "OneClassSVM": "One-Class Support Vector Machine",
    "IsolationForest": "Isolation Forest",
    "keras.Model": "Keras-based Variational Autoencoder",
}

st.set_page_config(page_title="BMS-Flux EDA Dashboard", layout="wide")
st.title("🔋 BMS-Flux Interactive EDA Dashboard")

# Auto-refresh every 15 minutes (900,000 ms)
st_autorefresh(interval=15 * 60 * 1000, key="data_refresh")

# Add CSS to constrain main container width
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }
    .element-container img, .element-container .js-plotly-plot {
        max-width: 100% !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper Functions ---
def get_all_runs():
    """Scan the 'runs' directory and return a sorted list of all run paths."""
    runs_path = Path("runs")
    if not runs_path.exists():
        return []
    # Ensure we only list directories, ignoring files like .DS_Store
    all_paths = runs_path.glob("*/*/*/*")
    return sorted(
        [p for p in all_paths if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def initialize_session_state(all_runs):
    """Initialize the session state on first run or if it's empty."""
    if "initialized" not in st.session_state and all_runs:
        latest_run = all_runs[0]
        st.session_state.selected_run = latest_run
        st.session_state.user_has_selected = False
        st.session_state.initialized = True
    elif "selected_run" not in st.session_state and all_runs:
        st.session_state.selected_run = all_runs[0]


# --- Main Logic ---
all_runs = get_all_runs()
if not all_runs:
    st.warning("No runs found. Please execute the pipeline to generate data.")
    st.stop()

# Initialize state and check if a newer run is available
initialize_session_state(all_runs)
latest_run_on_disk = all_runs[0]
if (
    not st.session_state.get("user_has_selected", False)
    and st.session_state.selected_run != latest_run_on_disk
):
    st.session_state.selected_run = latest_run_on_disk

# --- Sidebar ---
st.sidebar.header("Run Selector")


def on_run_change():
    st.session_state.user_has_selected = True


# Find the index of the currently selected run for the selectbox default
try:
    current_run_index = all_runs.index(st.session_state.selected_run)
except ValueError:
    current_run_index = 0  # Default to latest if not found

# Display runs using their path parts for readability
st.session_state.selected_run = st.sidebar.selectbox(
    "Select a Run",
    options=all_runs,
    format_func=lambda p: f"{p.parts[-4]}/{p.parts[-3]}/{p.parts[-2]}/{p.name}",
    index=current_run_index,
    on_change=on_run_change,
    key="run_selector",
)

if st.sidebar.button("Force Refresh"):
    st.cache_data.clear()
    st.rerun()

eda_folder = st.session_state.selected_run / "eda"
st.sidebar.info(f"Displaying data for run: **{st.session_state.selected_run.name}**")


# --- Data Loading ---
@st.cache_data
def load_data(path_to_csv):
    if not path_to_csv.exists():
        return None
    return pd.read_csv(path_to_csv)


main_data_path = eda_folder / "eda_data.csv"
main_data = load_data(main_data_path)

if main_data is None:
    st.error(
        f"EDA data not found for this run (`{main_data_path}`). Please select another run or check the pipeline logs."
    )
    st.stop()

# --- Main Dashboard Area ---
st.header(f"EDA Report: {st.session_state.selected_run.name}")

tab1, tab2, tab3 = st.tabs(
    ["Per-Cell Analytics", "Feature Exploration", "Static Reports"]
)

with tab1:
    st.subheader("Cell-Level Diagnostics")

    with st.expander("Per-Cell Summary Statistics", expanded=True):
        summary_path = eda_folder / "per_cell_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            st.dataframe(summary_df)
            st.download_button(
                "Download Summary",
                summary_df.to_csv(index=False),
                "per_cell_summary.csv",
            )
        else:
            st.warning("Per-cell summary not found.")

    with st.expander("Cell Balancing Over Time"):
        img_path = eda_folder / "cell_balancing_over_time.png"
        if img_path.exists():
            st.image(str(img_path))
        else:
            st.warning("Cell balancing plot not found.")

    with st.expander("Voltage/Temperature Excursions"):
        # Display excursion plots and dataframes
        for col in ["Voltage (V)", "Temperature (C)"]:
            exc_csv = eda_folder / f"{col}_excursions.csv"
            exc_img = eda_folder / f"{col}_excursions.png"
            if exc_csv.exists():
                st.write(f"**{col} Excursions Data**")
                df_exc = pd.read_csv(exc_csv)
                st.dataframe(df_exc.head())
                st.download_button(
                    f"Download {col} Excursions",
                    df_exc.to_csv(index=False),
                    f"{col}_excursions.csv",
                )
            if exc_img.exists():
                st.image(str(exc_img))

    # --- Anomaly Detection Section ---
    st.header("Anomaly Detection Analysis")
    anomaly_csv_path = st.session_state.selected_run / "anomaly" / "anomaly_results.csv"
    metrics_path = st.session_state.selected_run / "anomaly" / "metrics.json"
    model_info_path = st.session_state.selected_run / "anomaly" / "model_info.json"
    svm_model_file = (
        st.session_state.selected_run / "anomaly" / "anomaly_svm_model.joblib"
    )
    iforest_model_file = (
        st.session_state.selected_run / "anomaly" / "anomaly_iforest_model.joblib"
    )
    best_model_name = None
    best_model_file = None
    best_model_params = None
    if anomaly_csv_path.exists():
        anomaly_df = pd.read_csv(anomaly_csv_path)
        cell_ids = sorted(anomaly_df["Cell ID"].unique().tolist())
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("3D Anomaly Plot")
            anomaly_key = f"current_run_anomaly_cell_id_selector_{str(st.session_state.selected_run)}_anomaly"
            selected_cell_id = st.selectbox(
                "Select a Cell ID:", cell_ids, key=anomaly_key
            )
            plot_df = anomaly_df[anomaly_df["Cell ID"] == selected_cell_id]
            # Robustly select label column for coloring
            label_col = None
            for candidate in ["anomaly_label", "anomaly", "hybrid_anomaly"]:
                if candidate in plot_df.columns:
                    label_col = candidate
                    break
            if label_col is None:
                st.warning(
                    "No anomaly label column found in results. Cannot plot anomalies."
                )
            else:
                color_vals = plot_df[label_col]
                if color_vals.dtype in [int, float]:
                    color_vals = color_vals.map(
                        {0: "Normal", 1: "Anomaly", -1: "Anomaly"}
                    ).fillna("Normal")
                fig = px.scatter_3d(
                    plot_df,
                    x="Voltage (V)",
                    y="Current (A)",
                    z="Cell Temperature (°C)",
                    color=color_vals,
                    color_discrete_map={"Anomaly": "red", "Normal": "blue"},
                    title=f"Cell {selected_cell_id}",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Show Raw Data**")
            if st.checkbox(
                "Show Raw Anomaly Data for selected cell",
                key=f"show_raw_anomaly_current_run_{str(st.session_state.selected_run)}_anomaly",
            ):
                st.dataframe(plot_df)
            st.download_button(
                "Download Anomaly CSV",
                anomaly_csv_path.read_bytes(),
                file_name="anomaly_results.csv",
                key=f"current_run_anomaly_csv_{str(st.session_state.selected_run)}_anomaly",
            )
        with col2:
            st.subheader("Best Model & Metrics")
            # Load metrics and model info
            metrics = None
            model_info = None
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
            if model_info_path.exists():
                with open(model_info_path) as f:
                    model_info = json.load(f)
            # Determine best model (highest best_score)
            if model_info:
                scores = [
                    (k, v)
                    for k, v in model_info["params"].items()
                    if k.endswith("best_score")
                ]
                if scores:
                    best = max(scores, key=lambda x: x[1])
                    if "svm" in best[0]:
                        best_model_name = "OneClassSVM"
                        best_model_file = svm_model_file
                        best_model_params = model_info["params"]["svm"]
                    elif "iforest" in best[0]:
                        best_model_name = "IsolationForest"
                        best_model_file = iforest_model_file
                        best_model_params = model_info["params"]["iforest"]
            # Show best model card
            if best_model_name:
                st.markdown(
                    f"### {best_model_name} <span title='{MODEL_EXPLAIN.get(best_model_name, '')}' style='cursor:help;'>ℹ️</span>",
                    unsafe_allow_html=True,
                )
                if best_model_file is not None:
                    st.download_button(
                        f"Download {best_model_name} Model",
                        best_model_file.read_bytes(),
                        file_name=f"anomaly_{best_model_name.lower()}_model.joblib",
                    )
                if best_model_params is not None:
                    st.markdown("**Parameters:**")
                    for k, v in best_model_params.items():
                        info = METRIC_EXPLAIN.get(k, "")
                        st.markdown(
                            f"- <b>{k}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {v}",
                            unsafe_allow_html=True,
                        )
            # Show metrics as a table
            if metrics:
                st.markdown("#### Metrics")
                for k, v in metrics.items():
                    info = METRIC_EXPLAIN.get(k, "")
                    st.markdown(
                        f"- <b>{k}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {v}",
                        unsafe_allow_html=True,
                    )
            st.subheader("Anomaly Score Histogram")
            if "hybrid_anomaly_score" in anomaly_df.columns:
                fig_hist = px.histogram(
                    anomaly_df,
                    x="hybrid_anomaly_score",
                    nbins=30,
                    color="anomaly_label",
                    barmode="overlay",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Anomaly detection results not found for this run.")

    # --- Time Series Forecast Section ---
    st.header("Time Series Forecast Analysis")
    timeseries_folder = st.session_state.selected_run
    pred_path = timeseries_folder / "timeseries" / "timeseries_predictions.csv"
    timeseries_metrics_path = (
        timeseries_folder / "timeseries" / "timeseries_metrics.json"
    )
    timeseries_model_info_path = (
        timeseries_folder / "timeseries" / "timeseries_model_info.json"
    )
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        cell_ids = sorted(pred_df["Cell ID"].unique())
        targets = [
            col
            for col in pred_df.columns
            if col not in ["Cell ID", "Timestamp"] and not col.endswith("_pred")
        ]
        col1, col2 = st.columns([2, 3])
        with col1:
            ts_cell_key = f"current_run_ts_cell_id_selector_{str(st.session_state.selected_run)}_timeseries"
            ts_target_key = f"current_run_ts_target_selector_{str(st.session_state.selected_run)}_timeseries"
            ts_steps_key = f"current_run_ts_steps_slider_{str(st.session_state.selected_run)}_timeseries"
            selected_cell = st.selectbox("Select Cell ID:", cell_ids, key=ts_cell_key)
            selected_targets = st.multiselect(
                "Select Targets:", targets, default=targets[:1], key=ts_target_key
            )
            steps = st.slider("Prediction Steps:", 10, 30, 10, step=5, key=ts_steps_key)
            plot_df = pred_df[pred_df["Cell ID"] == selected_cell]
            fig = px.line()
            for target in selected_targets:
                fig.add_scatter(
                    x=plot_df["Timestamp"],
                    y=plot_df[target],
                    mode="lines+markers",
                    name=f"{target} (actual)",
                )
                pred_col = f"{target}_pred_{steps}"
                if pred_col in plot_df.columns:
                    fig.add_scatter(
                        x=plot_df["Timestamp"],
                        y=plot_df[pred_col],
                        mode="lines+markers",
                        name=f"{target} (pred {steps})",
                    )
            fig.update_layout(
                title=f"Forecast for Cell {selected_cell}",
                xaxis_title="Time",
                yaxis_title="Value",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download Time Series Predictions",
                pred_path.read_bytes(),
                file_name="timeseries_predictions.csv",
                key=f"current_run_ts_pred_csv_{str(st.session_state.selected_run)}_timeseries",
            )
        with col2:
            st.subheader("Best Model & Metrics")
            # Show best model and metrics in a card/table
            metrics = None
            model_info = None
            if timeseries_metrics_path.exists():
                with open(timeseries_metrics_path) as f:
                    metrics = json.load(f)
            if timeseries_model_info_path.exists():
                with open(timeseries_model_info_path) as f:
                    model_info = json.load(f)
            # Show best model (lowest RMSE or MAE)
            if model_info and metrics:

                def get_mae_safe(k):
                    v = metrics.get(k)
                    if isinstance(v, dict):
                        return v.get("mae", float("inf"))
                    return float("inf")

                best_key = min(metrics, key=get_mae_safe)
                if model_info is not None and best_key is not None:
                    best_model_name = model_info[best_key]
                st.markdown(
                    f"### {best_model_name} <span title='{MODEL_EXPLAIN.get(str(best_model_name), '')}' style='cursor:help;'>ℹ️</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Best Model for {best_key}:** {best_model_name}")
                st.markdown("**Metrics:**")
                if metrics is not None and best_key is not None:
                    value = metrics.get(best_key)
                    if isinstance(value, dict):
                        for mk, mv in value.items():
                            info = METRIC_EXPLAIN.get(str(mk), "")
                            st.markdown(
                                f"- <b>{mk}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {mv}",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No metrics found for this model.")
            else:
                st.info("No model info or metrics found for this run.")
    else:
        st.info("No time series predictions found for this run.")

    # --- Failure Prediction Section ---
    st.header("Failure Prediction Analysis")
    failure_metrics_path = (
        st.session_state.selected_run / "failure_prediction" / "failure_metrics.json"
    )
    failure_model_info_path = (
        st.session_state.selected_run / "failure_prediction" / "failure_model_info.json"
    )
    failure_pred_path = (
        st.session_state.selected_run / "failure_prediction" / "failure_predictions.csv"
    )
    if failure_metrics_path.exists():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("Confusion Matrix & ROC Curve")
            if failure_pred_path.exists():
                pred_df = pd.read_csv(failure_pred_path)
                y_true = pred_df["y_true"].replace({0: "No Failure", 1: "Failure"})
                y_pred = pred_df["y_pred"].replace({0: "No Failure", 1: "Failure"})
                cm = pd.crosstab(
                    y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]
                )
                st.write("Confusion Matrix:")
                st.dataframe(cm)
                if "y_prob" in pred_df.columns:
                    fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob"])
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(
                        x=fpr,
                        y=tpr,
                        title=f"ROC Curve (AUC={roc_auc:.2f})",
                        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                st.download_button(
                    "Download Failure Predictions",
                    failure_pred_path.read_bytes(),
                    file_name="failure_predictions.csv",
                    key=f"current_run_failure_pred_csv_{str(st.session_state.selected_run)}_failure",
                )
        with col2:
            st.subheader("Best Model & Metrics")
            metrics = None
            model_info = None
            if failure_metrics_path.exists():
                with open(failure_metrics_path) as f:
                    metrics = json.load(f)
            if failure_model_info_path.exists():
                with open(failure_model_info_path) as f:
                    model_info = json.load(f)
            # Show only the best model
            if model_info and "best_model" in model_info:
                best_model_raw = model_info["best_model"]
                best_model_name = None
                if isinstance(best_model_raw, str):
                    parts = best_model_raw.split("'")
                    if len(parts) >= 3:
                        best_model_name = parts[-2]
                    else:
                        best_model_name = best_model_raw
                else:
                    best_model_name = str(best_model_raw)
                st.markdown(
                    f"### {best_model_name} <span title='{MODEL_EXPLAIN.get(str(best_model_name), '')}' style='cursor:help;'>ℹ️</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Metrics:**")
                if metrics is not None and model_info.get("best_model") is not None:
                    for mk, mv in metrics.items():
                        info = METRIC_EXPLAIN.get(str(mk), "")
                        st.markdown(
                            f"- <b>{mk}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {mv}",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No best model info found for this run.")
    else:
        st.info("Failure prediction results not found for this run.")

with tab2:
    st.subheader("Interactive Feature Exploration")

    # Sidebar Filters for this tab
    st.sidebar.header("Feature Filters")
    num_cols = [
        c for c in main_data.columns if main_data[c].dtype in ["float64", "int64"]
    ]

    x_axis = st.sidebar.selectbox(
        "X-Axis",
        num_cols,
        index=num_cols.index("Timestamp") if "Timestamp" in num_cols else 0,
    )
    y_axis = st.sidebar.selectbox(
        "Y-Axis",
        num_cols,
        index=num_cols.index("Voltage (V)") if "Voltage (V)" in num_cols else 1,
    )
    color_by = st.sidebar.selectbox("Color By", ["Cell ID"] + num_cols, index=0)

    fig = px.scatter(
        main_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs. {x_axis}"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    dist_feat = st.selectbox("Select Feature for Distribution", num_cols)
    fig_dist = px.histogram(
        main_data,
        x=dist_feat,
        color="Cell ID",
        marginal="box",
        title=f"Distribution of {dist_feat}",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("Generated Static Plots & Reports")
    # Display all other PNGs from the EDA folder
    for img_file in sorted(eda_folder.glob("*.png")):
        if "cell_balancing" not in img_file.name and "excursions" not in img_file.name:
            with st.expander(
                img_file.name.replace("_", " ").replace(".png", "").title()
            ):
                st.image(str(img_file))

st.sidebar.markdown("---")
st.sidebar.info("Refresh the page to see the latest run after a pipeline execution.")

# --- Run Summary ---
st.title("BMS-Flux ML Ops Dashboard")
run_folder = st.session_state.get("selected_run", None)
if run_folder:
    st.info(f"**Run Folder:** {run_folder}")
    config_path = Path(run_folder) / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            st.markdown("**Config Used:**")
            st.code(f.read(), language="yaml")
    st.markdown("**Random Seed:** 42")
    # Download all artifacts
    if st.button("Download All Artifacts as ZIP"):
        zip_path = Path(run_folder) / "artifacts.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for folder, _, files in os.walk(run_folder):
                for file in files:
                    file_path = Path(folder) / file
                    zipf.write(file_path, arcname=file_path.relative_to(run_folder))
        with open(zip_path, "rb") as f:
            st.download_button("Download ZIP", f, file_name="artifacts.zip")
    # Rerun pipeline
    if st.button("Re-run Pipeline for this config"):
        subprocess.Popen(
            ["python", "agent-driver/run_full_pipeline.py"]
        )  # non-blocking
        st.success("Pipeline re-run started!")

# --- Compare Runs Tab ---
st.header("Compare Two Runs Side by Side")
runs_root = Path("runs")
all_runs = [
    p
    for p in runs_root.glob("**/*")
    if p.is_dir() and (p / "anomaly/anomaly_results.csv").exists()
]
run_options = [str(p) for p in all_runs]
selected_runs = st.multiselect(
    "Select two runs to compare:", run_options, max_selections=2
)

if len(selected_runs) != 2:
    st.info("Please select exactly two runs to compare.")
    st.stop()

run_left, run_right = selected_runs

cols = st.columns(2)
for idx, (run, col) in enumerate(zip([run_left, run_right], cols)):
    with col:
        st.markdown(f"### Run: {run}")
        # --- Anomaly Section ---
        metrics_path = Path(run) / "anomaly" / "metrics.json"
        anomaly_csv_path = Path(run) / "anomaly" / "anomaly_results.csv"
        if anomaly_csv_path.exists():
            anomaly_df = pd.read_csv(anomaly_csv_path)
            cell_ids = sorted(anomaly_df["Cell ID"].unique().tolist())
            anomaly_key = f"compare_{idx}_anomaly_cell_id_selector_{run}_anomaly"
            selected_cell_id = st.selectbox(
                "Select a Cell ID:", cell_ids, key=anomaly_key
            )
            plot_df = anomaly_df[anomaly_df["Cell ID"] == selected_cell_id]
            # Robustly select label column for coloring
            label_col = None
            for candidate in ["anomaly_label", "anomaly", "hybrid_anomaly"]:
                if candidate in plot_df.columns:
                    label_col = candidate
                    break
            if label_col is None:
                st.warning(
                    "No anomaly label column found in results. Cannot plot anomalies."
                )
            else:
                color_vals = plot_df[label_col]
                if color_vals.dtype in [int, float]:
                    color_vals = color_vals.map(
                        {0: "Normal", 1: "Anomaly", -1: "Anomaly"}
                    ).fillna("Normal")
                fig = px.scatter_3d(
                    plot_df,
                    x="Voltage (V)",
                    y="Current (A)",
                    z="Cell Temperature (°C)",
                    color=color_vals,
                    color_discrete_map={"Anomaly": "red", "Normal": "blue"},
                    title=f"Cell {selected_cell_id}",
                )
                st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download Anomaly CSV",
                anomaly_csv_path.read_bytes(),
                file_name=f"anomaly_results_{run}.csv",
                key=f"compare_{idx}_anomaly_csv_{run}_anomaly",
            )
            # Show metrics
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                st.markdown("#### Anomaly Metrics")
                if metrics is not None and run is not None:
                    for mk, mv in metrics.items():
                        info = METRIC_EXPLAIN.get(mk, "")
                        st.markdown(
                            f"- <b>{mk}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {mv}",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No metrics found for this run.")
        else:
            st.info("Anomaly detection results not found for this run.")
        # --- Time Series Section ---
        pred_path = Path(run) / "timeseries" / "timeseries_predictions.csv"
        timeseries_metrics_path = Path(run) / "timeseries" / "timeseries_metrics.json"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            cell_ids = sorted(pred_df["Cell ID"].unique())
            targets = [
                col
                for col in pred_df.columns
                if col not in ["Cell ID", "Timestamp"] and not col.endswith("_pred")
            ]
            ts_cell_key = f"compare_{idx}_ts_cell_id_selector_{run}_timeseries"
            ts_target_key = f"compare_{idx}_ts_target_selector_{run}_timeseries"
            ts_steps_key = f"compare_{idx}_ts_steps_slider_{run}_timeseries"
            selected_cell = st.selectbox(
                "Select Cell ID (TS):", cell_ids, key=ts_cell_key
            )
            selected_targets = st.multiselect(
                "Select Targets:", targets, default=targets[:1], key=ts_target_key
            )
            steps = st.slider("Prediction Steps:", 10, 30, 10, step=5, key=ts_steps_key)
            plot_df = pred_df[pred_df["Cell ID"] == selected_cell]
            fig = px.line()
            for target in selected_targets:
                fig.add_scatter(
                    x=plot_df["Timestamp"],
                    y=plot_df[target],
                    mode="lines+markers",
                    name=f"{target} (actual)",
                )
                pred_col = f"{target}_pred_{steps}"
                if pred_col in plot_df.columns:
                    fig.add_scatter(
                        x=plot_df["Timestamp"],
                        y=plot_df[pred_col],
                        mode="lines+markers",
                        name=f"{target} (pred {steps})",
                    )
            fig.update_layout(
                title=f"Forecast for Cell {selected_cell}",
                xaxis_title="Time",
                yaxis_title="Value",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "Download Time Series Predictions",
                pred_path.read_bytes(),
                file_name="timeseries_predictions.csv",
                key=f"compare_{idx}_ts_pred_csv_{run}_timeseries",
            )
            # Show metrics
            if timeseries_metrics_path.exists():
                with open(timeseries_metrics_path) as f:
                    metrics = json.load(f)
                st.markdown("#### Time Series Metrics")
                if metrics is not None and run is not None:
                    for mk, mv in metrics.items():
                        info = METRIC_EXPLAIN.get(mk, "")
                        st.markdown(
                            f"- <b>{mk}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {mv}",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("No time series metrics found for this run.")
        else:
            st.info("No time series predictions found for this run.")
        # --- Failure Prediction Section ---
        failure_metrics_path = Path(run) / "failure_prediction" / "failure_metrics.json"
        failure_pred_path = Path(run) / "failure_prediction" / "failure_predictions.csv"
        if failure_metrics_path.exists():
            st.subheader("Failure Prediction Analysis")
            if failure_pred_path.exists():
                pred_df = pd.read_csv(failure_pred_path)
                y_true = pred_df["y_true"].replace({0: "No Failure", 1: "Failure"})
                y_pred = pred_df["y_pred"].replace({0: "No Failure", 1: "Failure"})
                cm = pd.crosstab(
                    y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]
                )
                st.write("Confusion Matrix:")
                st.dataframe(cm)
                if "y_prob" in pred_df.columns:
                    fpr, tpr, _ = roc_curve(pred_df["y_true"], pred_df["y_prob"])
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(
                        x=fpr,
                        y=tpr,
                        title=f"ROC Curve (AUC={roc_auc:.2f})",
                        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                st.download_button(
                    "Download Failure Predictions",
                    failure_pred_path.read_bytes(),
                    file_name="failure_predictions.csv",
                    key=f"compare_{idx}_failure_pred_csv_{run}_failure",
                )
            # Show metrics
            with open(failure_metrics_path) as f:
                metrics = json.load(f)
            st.markdown("#### Failure Prediction Metrics")
            if metrics is not None and run is not None:
                for mk, mv in metrics.items():
                    info = METRIC_EXPLAIN.get(mk, "")
                    st.markdown(
                        f"- <b>{mk}</b> <span title='{info}' style='cursor:help;'>ℹ️</span>: {mv}",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Failure prediction results not found for this run.")
