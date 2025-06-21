import os
import pandas as pd
from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import evaluate_quality, get_column_plot

# --- Robustness Patch: Helper for agent-friendly, per-folder output ---
def _get_abs_path(base_folder, subfolder):
    abs_path = os.path.join(base_folder, subfolder)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def generate(folder_name: str):
    """
    Generate synthetic data using SDV for all CSVs in the specified folder.
    - Requires folder to have: metadata.json and one CSV per table described.
    - Output: 'synthetic_data' folder with synthetic CSVs.
    """
    # Check if the data folder exists
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"The folder {folder_name} does not exist.")

    # Check if metadata file exists
    metadata_file = os.path.join(folder_name, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"The metadata file {metadata_file} does not exist.")

    try:
        # Load CSV data files from the specified folder
        connector = CSVHandler()
        data = connector.read(folder_name=folder_name)

        # Load metadata
        metadata = Metadata.load_from_json(metadata_file)

        # Create and train synthesizer
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(data)

        # Generate synthetic data
        synthetic_data = synthesizer.sample(scale=1)

        # --- Robustness Patch: Save synthetic data to <folder_name>/synthetic_data ---
        synthetic_folder = _get_abs_path(folder_name, "synthetic_data")
        for table_name, df in synthetic_data.items():
            output_file = os.path.join(synthetic_folder, f"{table_name}.csv")
            df.to_csv(output_file, index=False)

        return f"Data generated successfully and saved in '{synthetic_folder}' folder with {len(synthetic_data)} tables named as {list(synthetic_data.keys())} CSV files."

    # Handle exceptions during data generation
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating synthetic data: {e}")

def evaluate(folder_name: str):
    """
    Evaluate synthetic data vs. real data using SDV.
    - Requires synthetic data in <folder_name>/synthetic_data and real data in folder_name.
    - Returns dict with overall score and per-table/column metrics.
    """
    # --- Robustness Patch: Use per-input-folder synthetic_data folder ---
    synthetic_folder = os.path.join(folder_name, "synthetic_data")

    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Real data folder not found: {folder_name}")
    if not os.path.exists(synthetic_folder):
        raise FileNotFoundError(
            f"Synthetic data folder not found in {synthetic_folder}. Please generate synthetic data first."
        )

    metadata_file = os.path.join(folder_name, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"The metadata file {metadata_file} does not exist.")

    try:
        metadata = Metadata.load_from_json(metadata_file)
        table_names = metadata.tables

        real_data_dict = {}
        synthetic_data_dict = {}
        for table_name in table_names:
            real_path = os.path.join(folder_name, f"{table_name}.csv")
            synthetic_path = os.path.join(synthetic_folder, f"{table_name}.csv")

            if not os.path.exists(real_path):
                raise FileNotFoundError(f"Real data file not found: {real_path}")
            if not os.path.exists(synthetic_path):
                raise FileNotFoundError(
                    f"Synthetic data file not found: {synthetic_path}"
                )

            real_data_dict[table_name] = pd.read_csv(real_path)
            synthetic_data_dict[table_name] = pd.read_csv(synthetic_path)

        quality_report = evaluate_quality(
            real_data=real_data_dict,
            synthetic_data=synthetic_data_dict,
            metadata=metadata,
            verbose=False,
        )

        overall_score = quality_report.get_score()
        properties_df = quality_report.get_properties()
        properties = properties_df.to_dict(orient="records")

        return {"Overall Score": overall_score, "Properties": properties}

    except Exception as e:
        raise RuntimeError(f"An error occurred during evaluation: {e}")

def visualize(
    folder_name: str,
    table_name: str,
    column_name: str,
    visualization_folder: str = "evaluation_plots",
):
    """
    Visualize the distribution of a specific column in a table for both real and synthetic data.
    - Requires synthetic data in <folder_name>/synthetic_data, real data in folder_name.
    - Output: PNG saved to <folder_name>/evaluation_plots.
    """
    synthetic_folder = os.path.join(folder_name, "synthetic_data")
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Real data folder not found: {folder_name}")
    if not os.path.exists(synthetic_folder):
        raise FileNotFoundError(
            "Synthetic data folder not found. Please generate synthetic data first."
        )

    metadata_file = os.path.join(folder_name, "metadata.json")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"The metadata file {metadata_file} does not exist.")

    try:
        metadata = Metadata.load_from_json(metadata_file)
        if table_name not in metadata.tables:
            raise ValueError(f"Table '{table_name}' not found in metadata")

        real_path = os.path.join(folder_name, f"{table_name}.csv")
        synthetic_path = os.path.join(synthetic_folder, f"{table_name}.csv")

        if not os.path.exists(real_path):
            raise FileNotFoundError(f"Real data file not found: {real_path}")
        if not os.path.exists(synthetic_path):
            raise FileNotFoundError(f"Synthetic data file not found: {synthetic_path}")

        real_data = pd.read_csv(real_path)
        synthetic_data = pd.read_csv(synthetic_path)

        if column_name not in real_data.columns:
            raise ValueError(
                f"Column '{column_name}' not found in table '{table_name}'"
            )

        real_data_dict = {table_name: real_data}
        synthetic_data_dict = {table_name: synthetic_data}

        # --- Robustness Patch: Save visualization inside <folder_name> for agent chaining ---
        abs_vis_folder = _get_abs_path(folder_name, visualization_folder)

        fig = get_column_plot(
            real_data=real_data_dict,
            synthetic_data=synthetic_data_dict,
            metadata=metadata,
            table_name=table_name,
            column_name=column_name,
        )

        if fig is None:
            raise ValueError(
                f"Could not generate visualization for {table_name}.{column_name}"
            )

        safe_column_name = column_name.replace(" ", "_").replace("/", "_")
        filename = f"{table_name}_{safe_column_name}.png"
        filepath = os.path.join(abs_vis_folder, filename)

        fig.write_image(filepath)
        return f"Visualization for {table_name}.{column_name} saved successfully at {os.path.abspath(filepath)}"
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during visualization: {e}")