# Energy Price Forecasting & Anomaly Detection

This project implements an end-to-end pipeline for forecasting energy prices (DK, ES, FR), detecting anomalies, and simulating real-time ingestion and monitoring.

## Project Structure

- `data/`: Contains cleaned CSV data files.
- `src/`: Source code for the pipeline.
  - `classicmodels/`: ARIMA/SARIMA implementations.
  - `neural/`: LSTM/GRU/RNN implementations + Encoder-Decoder with Attention.
  - `anomaly/`: Anomaly detection logic (ML-based).
  - `online/`: Real-time simulation and dashboard.
- `outputs/`: Stores models, metrics, plots, and simulation results.
- `notebooks/`: Jupyter notebooks for analysis and reporting.

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**:
   - Check `config.yaml` for file paths and simulation settings.

## Usage

### 1. Anomaly Detection (ML)

Train the anomaly classifier and generate silver labels:

```bash
# Process a single country
python -m src.anomaly.anomaly_ml --country DK --source ensemble

# Or process all available anomaly files
python -m src.anomaly.anomaly_ml --batch
```

### 2. Live Simulation

Run the online simulation to mimic real-time data ingestion and model updating:

```bash
# Run for a specific country (e.g., DK, ES, FR)
python -m src.online.simulation --country DK
```

### 3. Dashboard

Launch the interactive dashboard to visualize the simulation results:

```bash
streamlit run src/online/dashboard.py
```

## Results

- Metrics are stored in `outputs/metrics/`.
- Simulation logs and results are in `outputs/`.
- Anomaly labels and checkpoints are in `outputs/anomaly_ml/` and `outputs/checkpoints/`.
