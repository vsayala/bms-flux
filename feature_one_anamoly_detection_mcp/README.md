# Feature One: Anomaly Detection MCP Server

This MCP server provides hybrid anomaly detection tools for battery sensor data. It uses One-Class SVM, Isolation Forest, and Variational Autoencoder, with robust preprocessing and visualization.

## Features

- **Hybrid anomaly detection** using SVM, Isolation Forest, and VAE
- **Advanced feature engineering** and imputation
- **3D anomaly visualization** for voltage, current, temperature
- Exposes tools via the MCP protocol for easy orchestration

## Usage

### Install Dependencies

```bash
uv sync
```

### Run the MCP Server

```bash
uv run --with mcp server.py
```

### Example Tools

- `detect_anomalies(data_path: str, chunksize: int = 100000) -> str`
  Runs the full anomaly detection pipeline on your battery data CSV.

- `visualize_anomalies_3d(result_csv: str, ...) -> str`
  Visualizes the anomalies found in 3D (Voltage, Current, Temperature).

## Customization

You may edit `tools.py` to add models, new feature engineering, or metrics.

## MCP Integration

To use with an MCP host, add an entry like this to your `mcp.json`:

```json
{
  "mcpServers": {
    "feature_one_anomaly_detection": {
      "command": "uv",
      "args": [
        "--directory",
        "./feature-one-anamoly-detection-mcp",
        "run",
        "--with",
        "mcp",
        "server.py"
      ]
    }
  }
}
```

## License

MIT
