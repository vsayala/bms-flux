import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="BMS-Flux EDA Dashboard", layout="wide")
st.title("🔋 BMS-Flux Interactive EDA Dashboard")

# Auto-refresh every 15 minutes (900,000 ms)
st_autorefresh(interval=15 * 60 * 1000, key="data_refresh")


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
