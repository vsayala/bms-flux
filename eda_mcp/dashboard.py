import streamlit as st
import pandas as pd
import os
import glob
from pathlib import Path
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="BMS-Flux EDA Dashboard", layout="wide")
st.title("🔋 BMS-Flux Interactive EDA Dashboard")

# --- Run Selector ---
st.sidebar.header("Run Selector")

runs_path = Path("runs")
years = sorted([d.name for d in runs_path.iterdir() if d.is_dir()], reverse=True)
if not years:
    st.warning("No runs found in the 'runs' directory.")
    st.stop()

selected_year = st.sidebar.selectbox("Year", years)
months = sorted([d.name for d in (runs_path / selected_year).iterdir() if d.is_dir()], reverse=True)
selected_month = st.sidebar.selectbox("Month", months)
days = sorted([d.name for d in (runs_path / selected_year / selected_month).iterdir() if d.is_dir()], reverse=True)
selected_day = st.sidebar.selectbox("Day", days)
run_ids = sorted([d.name for d in (runs_path / selected_year / selected_month / selected_day).iterdir() if d.is_dir()], reverse=True)
selected_run_id = st.sidebar.selectbox("Run ID", run_ids)

run_folder = runs_path / selected_year / selected_month / selected_day / selected_run_id
eda_folder = run_folder / "eda"
st.sidebar.info(f"Displaying data for run: **{selected_run_id}**")

# --- Data Loading ---
@st.cache_data
def load_data(path_to_csv):
    if not path_to_csv.exists():
        return None
    return pd.read_csv(path_to_csv)

main_data_path = eda_folder / "eda_data.csv"
main_data = load_data(main_data_path)

if main_data is None:
    st.error(f"EDA data not found for this run (`{main_data_path}`). Please select another run or check the pipeline logs.")
    st.stop()

# --- Main Dashboard Area ---
st.header(f"EDA Report: {selected_run_id}")

tab1, tab2, tab3 = st.tabs(["Per-Cell Analytics", "Feature Exploration", "Static Reports"])

with tab1:
    st.subheader("Cell-Level Diagnostics")
    
    with st.expander("Per-Cell Summary Statistics", expanded=True):
        summary_path = eda_folder / "per_cell_summary.csv"
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            st.dataframe(summary_df)
            st.download_button("Download Summary", summary_df.to_csv(index=False), "per_cell_summary.csv")
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
                st.download_button(f"Download {col} Excursions", df_exc.to_csv(index=False), f"{col}_excursions.csv")
            if exc_img.exists():
                 st.image(str(exc_img))

with tab2:
    st.subheader("Interactive Feature Exploration")
    
    # Sidebar Filters for this tab
    st.sidebar.header("Feature Filters")
    num_cols = [c for c in main_data.columns if main_data[c].dtype in ['float64', 'int64']]
    
    x_axis = st.sidebar.selectbox("X-Axis", num_cols, index=num_cols.index("Timestamp") if "Timestamp" in num_cols else 0)
    y_axis = st.sidebar.selectbox("Y-Axis", num_cols, index=num_cols.index("Voltage (V)") if "Voltage (V)" in num_cols else 1)
    color_by = st.sidebar.selectbox("Color By", ["Cell ID"] + num_cols, index=0)

    fig = px.scatter(main_data, x=x_axis, y=y_axis, color=color_by, title=f"{y_axis} vs. {x_axis}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Distributions")
    dist_feat = st.selectbox("Select Feature for Distribution", num_cols)
    fig_dist = px.histogram(main_data, x=dist_feat, color="Cell ID", marginal="box", title=f"Distribution of {dist_feat}")
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("Generated Static Plots & Reports")
    # Display all other PNGs from the EDA folder
    for img_file in sorted(eda_folder.glob("*.png")):
        if "cell_balancing" not in img_file.name and "excursions" not in img_file.name:
            with st.expander(img_file.name.replace("_", " ").replace(".png", "").title()):
                st.image(str(img_file))

st.sidebar.markdown("---")
st.sidebar.info("Refresh the page to see the latest run after a pipeline execution.") 