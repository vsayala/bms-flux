import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jinja2
import pdfkit
import base64
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# Helper for master log
MASTER_LOG = "bms_master.log"


def master_log(msg):
    with open(MASTER_LOG, "a") as f:
        f.write(f"[EDA] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def run_eda(data_path: str, run_folder: Path) -> dict:
    # 1. Setup EDA-specific output folder and logger
    eda_folder = run_folder / "eda"
    eda_folder.mkdir(exist_ok=True)

    log_path = eda_folder / "eda.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

    logging.info(f"Starting EDA. Outputs will be saved to: {eda_folder}")
    master_log(f"Starting EDA. Outputs will be saved to: {eda_folder}")

    outputs = []

    try:
        df = pd.read_csv(data_path)
        # Save a copy of the data used for EDA in the run folder for the dashboard
        df.to_csv(eda_folder / "eda_data.csv", index=False)
        logging.info(f"Loaded data: {data_path} (shape={df.shape})")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        master_log(f"Failed to load data: {e}")
        return {"error": str(e)}

    # --- 1. Per-Cell Analysis ---
    try:
        if "Cell ID" in df.columns:
            cell_groups = df.groupby("Cell ID")
            # Per-cell summary
            per_cell_stats_path = eda_folder / "per_cell_summary.csv"
            cell_groups.describe().stack(level=0).to_csv(per_cell_stats_path)
            outputs.append(str(per_cell_stats_path))
            logging.info("Saved per-cell summary stats.")
            master_log("Saved per-cell summary stats.")
            # Cell balancing: voltage spread over time
            if "Voltage (V)" in df.columns and "Timestamp" in df.columns:
                pivot = df.pivot(
                    index="Timestamp", columns="Cell ID", values="Voltage (V)"
                )
                plt.figure(figsize=(12, 6))
                plt.plot(pivot, alpha=0.7)
                plt.title("Cell Voltage Over Time (Balancing)")
                plt.xlabel("Timestamp")
                plt.ylabel("Voltage (V)")
                plt.legend(
                    pivot.columns,
                    title="Cell ID",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
                plt.tight_layout()
                cell_bal_path = eda_folder / "cell_balancing_over_time.png"
                plt.savefig(cell_bal_path)
                plt.close()
                outputs.append(str(cell_bal_path))
                logging.info("Saved cell balancing plot.")
                master_log("Saved cell balancing plot.")
                # Heatmap
                plt.figure(figsize=(14, 6))
                sns.heatmap(pivot.T, cmap="viridis", cbar_kws={"label": "Voltage (V)"})
                plt.title("Cell Voltage Heatmap (Cell x Time)")
                plt.xlabel("Timestamp")
                plt.ylabel("Cell ID")
                plt.tight_layout()
                heatmap_path = eda_folder / "cell_voltage_heatmap.png"
                plt.savefig(heatmap_path)
                plt.close()
                outputs.append(str(heatmap_path))
                logging.info("Saved cell voltage heatmap.")
                master_log("Saved cell voltage heatmap.")
    except Exception as e:
        logging.warning(f"Per-cell analysis failed: {e}")
        master_log(f"Per-cell analysis failed: {e}")

    # --- 2. Failure/Anomaly by Cell ---
    try:
        label_col = None
        for col in df.columns:
            if col.lower() in ["failure", "anomaly", "label", "target"]:
                label_col = col
                break
        if label_col and "Cell ID" in df.columns:
            fail_rate = df.groupby("Cell ID")[label_col].mean()
            fail_rate_path = eda_folder / "failure_rate_by_cell.csv"
            fail_rate.to_csv(fail_rate_path)
            outputs.append(str(fail_rate_path))
            plt.figure(figsize=(8, 4))
            fail_rate.plot(kind="bar")
            plt.title(f"{label_col} Rate by Cell")
            plt.xlabel("Cell ID")
            plt.ylabel(f"Mean {label_col}")
            plt.tight_layout()
            fail_rate_plot = eda_folder / "failure_rate_by_cell.png"
            plt.savefig(fail_rate_plot)
            plt.close()
            outputs.append(str(fail_rate_plot))
            logging.info("Saved failure/anomaly rate by cell.")
            master_log("Saved failure/anomaly rate by cell.")
            # Feature importance for failure
            X = (
                df.select_dtypes(include=[np.number])
                .drop(columns=[label_col], errors="ignore")
                .fillna(0)
            )
            y = df[label_col]
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns)
            imp_path = eda_folder / "feature_importance_failure.csv"
            importances.sort_values(ascending=False).to_csv(imp_path)
            outputs.append(str(imp_path))
            plt.figure(figsize=(8, 4))
            importances.sort_values(ascending=False).head(15).plot(kind="bar")
            plt.title(f"Feature Importance for {label_col}")
            plt.tight_layout()
            fi_plot = eda_folder / "feature_importance_failure.png"
            plt.savefig(fi_plot)
            plt.close()
            outputs.append(str(fi_plot))
            logging.info("Saved feature importance for failure/anomaly.")
            master_log("Saved feature importance for failure/anomaly.")
    except Exception as e:
        logging.warning(f"Failure/anomaly by cell failed: {e}")
        master_log(f"Failure/anomaly by cell failed: {e}")

    # --- 3. Categorical/Binary/Continuous Feature Handling ---
    try:
        for col in df.columns:
            if df[col].dtype == bool or (
                df[col].nunique() <= 5 and df[col].dtype in [np.int64, np.float64]
            ):
                plt.figure(figsize=(6, 4))
                df[col].value_counts().plot(kind="bar")
                plt.title(f"Bar Plot of {col}")
                plt.tight_layout()
                bar_path = eda_folder / f"bar_{col}.png"
                plt.savefig(bar_path)
                plt.close()
                outputs.append(str(bar_path))
                logging.info(f"Saved bar plot for {col}.")
                master_log(f"Saved bar plot for {col}.")
            elif df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 5:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                dist_path = eda_folder / f"dist_{col}.png"
                plt.savefig(dist_path)
                plt.close()
                outputs.append(str(dist_path))
                logging.info(f"Saved distribution plot for {col}.")
                master_log(f"Saved distribution plot for {col}.")
    except Exception as e:
        logging.warning(f"Feature type handling failed: {e}")
        master_log(f"Feature type handling failed: {e}")

    # --- 4. Per-Cell Outlier and Drift Detection ---
    try:
        if "Cell ID" in df.columns and "Voltage (V)" in df.columns:
            cell_groups = df.groupby("Cell ID")
            outlier_stats = cell_groups["Voltage (V)"].agg(
                ["mean", "std", "min", "max"]
            )
            outlier_stats_path = eda_folder / "cell_voltage_stats.csv"
            outlier_stats.to_csv(outlier_stats_path)
            outputs.append(str(outlier_stats_path))
            logging.info("Saved per-cell voltage stats.")
            master_log("Saved per-cell voltage stats.")
    except Exception as e:
        logging.warning(f"Outlier/drift detection failed: {e}")
        master_log(f"Outlier/drift detection failed: {e}")

    # --- 5. More Domain Diagnostics ---
    try:
        # 1. SOC/SOH Drift Detection
        for drift_col in [c for c in df.columns if c.lower() in ["soc", "soh"]]:
            if "Cell ID" in df.columns and "Timestamp" in df.columns:
                pivot = df.pivot(index="Timestamp", columns="Cell ID", values=drift_col)
                plt.figure(figsize=(12, 6))
                plt.plot(pivot, alpha=0.7)
                plt.title(f"{drift_col} Drift Over Time by Cell")
                plt.xlabel("Timestamp")
                plt.ylabel(drift_col)
                plt.legend(
                    pivot.columns,
                    title="Cell ID",
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                )
                plt.tight_layout()
                drift_path = eda_folder / f"{drift_col}_drift_over_time.png"
                plt.savefig(drift_path)
                plt.close()
                outputs.append(str(drift_path))
                logging.info(f"Saved {drift_col} drift plot.")
                master_log(f"Saved {drift_col} drift plot.")
                # Drift stats
                drift_stats = pivot.diff().abs().mean()
                drift_stats_path = eda_folder / f"{drift_col}_drift_stats.csv"
                drift_stats.to_csv(drift_stats_path)
                outputs.append(str(drift_stats_path))
        # 2. Voltage/Temperature Excursions
        for exc_col, normal_range in [
            ("Voltage (V)", (2.5, 4.2)),
            ("Temperature (C)", (10, 60)),
        ]:
            if exc_col in df.columns and "Cell ID" in df.columns:
                excursions = df[
                    (df[exc_col] < normal_range[0]) | (df[exc_col] > normal_range[1])
                ]
                exc_path = eda_folder / f"{exc_col}_excursions.csv"
                excursions.to_csv(exc_path, index=False)
                outputs.append(str(exc_path))
                plt.figure(figsize=(10, 4))
                for cell, group in excursions.groupby("Cell ID"):
                    plt.scatter(
                        group["Timestamp"], group[exc_col], label=f"Cell {cell}", s=10
                    )
                plt.title(f"{exc_col} Excursions by Cell")
                plt.xlabel("Timestamp")
                plt.ylabel(exc_col)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()
                exc_plot = eda_folder / f"{exc_col}_excursions.png"
                plt.savefig(exc_plot)
                plt.close()
                outputs.append(str(exc_plot))
                logging.info(f"Saved {exc_col} excursions plot and CSV.")
                master_log(f"Saved {exc_col} excursions plot and CSV.")
        # 3. Cycle Life/Degradation Analysis
        for cyc_col in [
            c for c in df.columns if c.lower() in ["cycle", "age", "cycle_count"]
        ]:
            if "Capacity (Ah)" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.scatter(df[cyc_col], df["Capacity (Ah)"], alpha=0.5)
                plt.title(f"Capacity Fade vs {cyc_col}")
                plt.xlabel(cyc_col)
                plt.ylabel("Capacity (Ah)")
                plt.tight_layout()
                fade_path = eda_folder / f"capacity_fade_vs_{cyc_col}.png"
                plt.savefig(fade_path)
                plt.close()
                outputs.append(str(fade_path))
                logging.info(f"Saved capacity fade plot for {cyc_col}.")
                master_log(f"Saved capacity fade plot for {cyc_col}.")
            if "Resistance (mOhm)" in df.columns:
                plt.figure(figsize=(8, 4))
                plt.scatter(df[cyc_col], df["Resistance (mOhm)"], alpha=0.5)
                plt.title(f"Resistance Growth vs {cyc_col}")
                plt.xlabel(cyc_col)
                plt.ylabel("Resistance (mOhm)")
                plt.tight_layout()
                res_path = eda_folder / f"resistance_growth_vs_{cyc_col}.png"
                plt.savefig(res_path)
                plt.close()
                outputs.append(str(res_path))
                logging.info(f"Saved resistance growth plot for {cyc_col}.")
                master_log(f"Saved resistance growth plot for {cyc_col}.")
        # 4. Alarm/Event Analysis
        for alarm_col in [
            c for c in df.columns if "alarm" in c.lower() or "event" in c.lower()
        ]:
            plt.figure(figsize=(8, 4))
            df[alarm_col].value_counts().plot(kind="bar")
            plt.title(f"{alarm_col} Frequency")
            plt.tight_layout()
            alarm_path = eda_folder / f"alarm_{alarm_col}_freq.png"
            plt.savefig(alarm_path)
            plt.close()
            outputs.append(str(alarm_path))
            logging.info(f"Saved alarm/event plot for {alarm_col}.")
            master_log(f"Saved alarm/event plot for {alarm_col}.")
    except Exception as e:
        logging.warning(f"Domain diagnostics failed: {e}")
        master_log(f"Domain diagnostics failed: {e}")

    # --- 6. Generate HTML/PDF report as before (reuse previous code) ---
    try:
        template_str = """
        <html>
        <head><title>EDA Report - {{ job_run_id }}</title></head>
        <body>
        <h1>EDA Report</h1>
        <h2>Run ID: {{ job_run_id }}</h2>
        <h3>Per-Cell Summary</h3>
        <pre>{{ per_cell_stats }}</pre>
        <h3>Failure/Anomaly Rate by Cell</h3>
        <pre>{{ fail_rate }}</pre>
        <h3>Cell Voltage Stats</h3>
        <pre>{{ outlier_stats }}</pre>
        <h3>Plots</h3>
        {% for plot in plot_files %}
            <div><img src="data:image/png;base64,{{ plot }}" style="max-width:700px;"></div>
        {% endfor %}
        <h3>Interactive Plots</h3>
        {% for html in html_files %}
            <iframe src="{{ html }}" width="700" height="500"></iframe>
        {% endfor %}
        </body>
        </html>
        """

        def read_csv_txt(path):
            try:
                return open(path).read()
            except FileNotFoundError:
                return "(not found)"

        per_cell_stats = read_csv_txt(eda_folder / "per_cell_summary.csv")
        fail_rate = read_csv_txt(eda_folder / "failure_rate_by_cell.csv")
        outlier_stats = read_csv_txt(eda_folder / "cell_voltage_stats.csv")
        plot_files = []
        for f in eda_folder.iterdir():
            if f.suffix == ".png":
                with open(f, "rb") as imgf:
                    plot_files.append(base64.b64encode(imgf.read()).decode())
        html_files = [f.name for f in eda_folder.iterdir() if f.suffix == ".html"]
        template = jinja2.Template(template_str)
        html_report = template.render(
            job_run_id=run_folder.name,
            per_cell_stats=per_cell_stats,
            fail_rate=fail_rate,
            outlier_stats=outlier_stats,
            plot_files=plot_files,
            html_files=html_files,
        )
        html_path = eda_folder / "eda_report.html"
        with open(html_path, "w") as f:
            f.write(html_report)
        outputs.append(str(html_path))
        pdf_path = eda_folder / "eda_report.pdf"
        try:
            pdfkit.from_file(html_path, pdf_path)
            outputs.append(str(pdf_path))
        except Exception as e:
            logging.warning(f"PDF report generation failed: {e}")
            master_log(f"PDF report generation failed: {e}")
    except Exception as e:
        logging.warning(f"HTML report generation failed: {e}")
        master_log(f"HTML report generation failed: {e}")

    # --- 7. Save a log of all outputs ---
    with open(eda_folder / "eda_outputs.txt", "w") as f:
        for out in outputs:
            f.write(f"{out}\n")

    logging.info(f"EDA complete. {len(outputs)} outputs generated.")
    master_log(f"EDA complete. {len(outputs)} outputs generated in {eda_folder}")

    return {"eda_folder": str(eda_folder), "outputs": outputs}
