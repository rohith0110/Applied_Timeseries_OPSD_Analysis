"""
Train & evaluate neural models:
 - ANN, RNN, GRU, LSTM, BiLSTM

Expect config.yaml to provide:
 - cleaned_files: list of per-country cleaned CSVs with columns: timestamp, load, [exog...]
 - model training hyperparameters (epochs, batch_size, lr, seed)

Outputs:
 - outputs/forecasts/<CC>_NN_<model>_dev.csv
 - outputs/forecasts/<CC>_NN_<model>_test.csv
 - outputs/metrics/<CC>_NN_<model>_metrics.csv
"""

import os
import random
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ..metrics.metrics import compute_all_metrics


from .neural_models import (
    ANNForecaster,
    RNNForecaster,
    GRUForecaster,
    LSTMForecaster,
    BiLSTMForecaster,
)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32).view(1, -1, 1)
        )

    def forward(self, preds, target):

        target = target.unsqueeze(1)
        errors = target - preds
        loss = torch.maximum((self.quantiles - 1) * errors, self.quantiles * errors)
        return loss.mean()


def extract_quantile_arrays(preds, quantiles, lower_q, upper_q):
    q_arr = np.asarray(quantiles)
    median_idx = int(np.argmin(np.abs(q_arr - 0.5)))
    lower_idx = int(np.argmin(np.abs(q_arr - lower_q)))
    upper_idx = int(np.argmin(np.abs(q_arr - upper_q)))
    preds_point = preds[:, median_idx, :]
    preds_lower = preds[:, lower_idx, :]
    preds_upper = preds[:, upper_idx, :]
    return preds_point, preds_lower, preds_upper


def to_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SeqDataset(Dataset):
    def __init__(self, series, exog=None, input_window=168, horizon=24):
        """
        series: 1D numpy array of load
        exog: numpy array shape (n, k) or None
        We create sliding windows (X: last input_window -> y: next horizon vector)
        """
        self.series = np.asarray(series, dtype=np.float32)
        self.exog = np.asarray(exog, dtype=np.float32) if exog is not None else None
        self.T = input_window
        self.H = horizon
        self.indices = []
        n = len(self.series)

        for start in range(0, n - self.T - self.H + 1):
            self.indices.append(start)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.series[i : i + self.T].reshape(self.T, 1)
        y = self.series[i + self.T : i + self.T + self.H]
        if self.exog is not None:
            xe = self.exog[i : i + self.T]
            return x, xe, y
        else:
            return x, None, y


def collate_fn(batch):
    xs, xes, ys = [], [], []
    for item in batch:
        if len(item) == 3:
            x, xe, y = item
        else:
            x, y = item
            xe = None
        xs.append(x)
        ys.append(y)
        xes.append(xe)
    xs = torch.tensor(np.stack(xs))
    ys = torch.tensor(np.stack(ys))
    if xes[0] is not None:
        xes = torch.tensor(np.stack(xes))
    else:
        xes = None
    return xs, xes, ys


def train_one(model, loader, opt, device, loss_fn):
    model.train()
    total_loss = 0.0
    for x, xe, y in loader:
        x = x.to(device)
        y = y.to(device)
        if xe is not None:
            xe = xe.to(device)
        opt.zero_grad()
        pred = model(x, xe)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def eval_one(model, loader, device, loss_fn):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, xe, y in loader:
            x = x.to(device)
            y = y.to(device)
            if xe is not None:
                xe = xe.to(device)
            pred = model(x, xe)
            total_loss += loss_fn(pred, y).item() * x.size(0)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    if len(preds) == 0:
        return None, None, np.nan
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return preds, trues, total_loss / len(loader.dataset)


def run_experiment_for_country(csv_path, cfg, model_name, device):
    """
    Loads country CSV -> builds datasets -> trains model -> backtests (dev/test outputs)
    """
    df = (
        pd.read_csv(csv_path, parse_dates=["timestamp"])
        .set_index("timestamp")
        .asfreq("h")
    )

    df["load"] = df["load"].interpolate()

    exog_cols = cfg.get("exog_columns", [])
    exog_data = df[exog_cols].interpolate().values if exog_cols else None

    y = df["load"].values.astype(np.float32)
    n = len(y)

    train_end = int(0.8 * n)
    dev_end = int(0.9 * n)

    train_y = y[:train_end]
    dev_y = y[train_end:dev_end]
    test_y = y[dev_end:]

    input_window = to_int(cfg.get("nn_input_window", 168), 168)
    horizon = to_int(cfg.get("horizon", 24), 24)
    batch_size = to_int(cfg.get("batch_size", 128), 128)
    epochs = to_int(cfg.get("epochs", 20), 20)
    lr = to_float(cfg.get("lr", 1e-3), 1e-3)
    seed = to_int(cfg.get("seed", 42), 42)
    use_quantiles = bool(cfg.get("nn_use_quantiles", False))
    quantile_levels = None
    pi_lower = float(cfg.get("nn_pi_lower", 0.1))
    pi_upper = float(cfg.get("nn_pi_upper", 0.9))
    if use_quantiles:
        raw_q = cfg.get("nn_quantiles", [0.1, 0.5, 0.9])
        quantile_levels = sorted({float(q) for q in raw_q})
        if not any(abs(q - 0.5) < 1e-6 for q in quantile_levels):
            quantile_levels.append(0.5)
            quantile_levels = sorted(quantile_levels)

        pi_lower = max(min(pi_lower, 0.49), 0.0)
        pi_upper = min(max(pi_upper, 0.51), 1.0)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    scaler = StandardScaler()
    scaler.fit(train_y.reshape(-1, 1))
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()

    if exog_data is not None:

        exog_scalers = []
        X_scaled = np.zeros_like(exog_data, dtype=np.float32)
        for j in range(exog_data.shape[1]):
            sc = StandardScaler()
            sc.fit(exog_data[:train_end, j].reshape(-1, 1))
            exog_scalers.append(sc)
            X_scaled[:, j] = sc.transform(exog_data[:, j].reshape(-1, 1)).flatten()
    else:
        X_scaled = None

    ds = SeqDataset(y_scaled, exog=X_scaled, input_window=input_window, horizon=horizon)

    timestamps = df.index

    train_indices, dev_indices, test_indices = [], [], []
    for idx_start in ds.indices:
        target_start = idx_start + input_window
        target_end = target_start + horizon - 1

        ts0 = timestamps[target_start]
        if ts0 < timestamps[train_end]:
            train_indices.append(idx_start)
        elif ts0 < timestamps[dev_end]:
            dev_indices.append(idx_start)
        else:
            test_indices.append(idx_start)

    def make_loader(indices, shuffle=False):
        subset = torch.utils.data.Subset(ds, [ds.indices.index(i) for i in indices])
        return DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
        )

    train_loader = make_loader(train_indices, shuffle=True)
    dev_loader = make_loader(dev_indices, shuffle=False)
    test_loader = make_loader(test_indices, shuffle=False)

    exog_dim = X_scaled.shape[1] if X_scaled is not None else 0
    input_dim = 1
    if model_name.lower() == "ann":
        model = ANNForecaster(
            in_window=input_window,
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_sizes=(512, 256),
            out_h=horizon,
            quantile_levels=quantile_levels,
        )
    elif model_name.lower() == "rnn":
        model = RNNForecaster(
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_size=256,
            n_layers=1,
            out_h=horizon,
            quantile_levels=quantile_levels,
        )
    elif model_name.lower() == "gru":
        model = GRUForecaster(
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_size=256,
            n_layers=1,
            out_h=horizon,
            quantile_levels=quantile_levels,
        )
    elif model_name.lower() == "lstm":
        model = LSTMForecaster(
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_size=256,
            n_layers=1,
            out_h=horizon,
            quantile_levels=quantile_levels,
        )
    elif model_name.lower() == "bilstm":
        model = BiLSTMForecaster(
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_size=256,
            n_layers=1,
            out_h=horizon,
            quantile_levels=quantile_levels,
        )
    else:
        raise ValueError(f"Unknown model {model_name}")

    device = (
        device
        if torch.cuda.is_available() and device == "cuda"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if use_quantiles:
        loss_fn = QuantileLoss(quantile_levels).to(device)
    else:
        loss_fn = nn.MSELoss()

    best_dev = np.inf
    best_state = None
    patience = 5
    cur_pat = 0

    for ep in range(epochs):
        train_loss = train_one(model, train_loader, opt, device, loss_fn)
        preds_dev, trues_dev, dev_loss = eval_one(model, dev_loader, device, loss_fn)
        print(
            f"[{Path(csv_path).stem}] {model_name} epoch {ep+1}/{epochs} train_loss={train_loss:.6f} dev_loss={dev_loss:.6f}"
        )
        if dev_loss < best_dev:
            best_dev = dev_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            cur_pat = 0
        else:
            cur_pat += 1
            if cur_pat >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict({k: best_state[k].to(device) for k in best_state})

    preds_dev, trues_dev, _ = eval_one(model, dev_loader, device, loss_fn)
    preds_test, trues_test, _ = eval_one(model, test_loader, device, loss_fn)

    if use_quantiles:
        preds_dev_point, preds_dev_lower, preds_dev_upper = extract_quantile_arrays(
            preds_dev, quantile_levels, pi_lower, pi_upper
        )
        preds_test_point, preds_test_lower, preds_test_upper = extract_quantile_arrays(
            preds_test, quantile_levels, pi_lower, pi_upper
        )
    else:
        preds_dev_point, preds_dev_lower, preds_dev_upper = preds_dev, None, None
        preds_test_point, preds_test_lower, preds_test_upper = preds_test, None, None

    def inv_scale(preds_scaled):
        if preds_scaled is None:
            return None
        return scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(
            preds_scaled.shape
        )

    preds_dev_inv = inv_scale(preds_dev_point)
    trues_dev_inv = inv_scale(trues_dev)
    preds_test_inv = inv_scale(preds_test_point)
    trues_test_inv = inv_scale(trues_test)
    lower_dev_inv = inv_scale(preds_dev_lower)
    upper_dev_inv = inv_scale(preds_dev_upper)
    lower_test_inv = inv_scale(preds_test_lower)
    upper_test_inv = inv_scale(preds_test_upper)

    def build_forecast_df(
        indices, preds_inv, trues_inv, lower_inv=None, upper_inv=None
    ):
        rows = []
        has_intervals = lower_inv is not None and upper_inv is not None
        for i_idx, start in enumerate(indices):
            train_end_idx = start + input_window - 1
            train_end_ts = timestamps[train_end_idx]
            for h in range(horizon):
                ts = timestamps[start + input_window + h]
                lo_val = float(lower_inv[i_idx, h]) if has_intervals else np.nan
                hi_val = float(upper_inv[i_idx, h]) if has_intervals else np.nan
                rows.append(
                    {
                        "timestamp": ts,
                        "train_end": train_end_ts,
                        "horizon": h + 1,
                        "y_true": float(trues_inv[i_idx, h]),
                        "yhat": float(preds_inv[i_idx, h]),
                        "lo": lo_val,
                        "hi": hi_val,
                    }
                )
        return pd.DataFrame(rows)

    dev_df = build_forecast_df(
        dev_indices, preds_dev_inv, trues_dev_inv, lower_dev_inv, upper_dev_inv
    )
    test_df = build_forecast_df(
        test_indices, preds_test_inv, trues_test_inv, lower_test_inv, upper_test_inv
    )

    out_forecasts_dir = Path(cfg["output_dir"]) / "forecasts"
    out_metrics_dir = Path(cfg["output_dir"]) / "metrics"
    out_forecasts_dir.mkdir(parents=True, exist_ok=True)
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    country = Path(csv_path).stem
    dev_df.to_csv(out_forecasts_dir / f"{country}_NN_{model_name}_dev.csv", index=False)
    test_df.to_csv(
        out_forecasts_dir / f"{country}_NN_{model_name}_test.csv", index=False
    )

    def compute_metrics(df_fore):
        lower = df_fore["lo"].values if "lo" in df_fore else None
        upper = df_fore["hi"].values if "hi" in df_fore else None
        return compute_all_metrics(
            df_fore["y_true"].values,
            df_fore["yhat"].values,
            lower=lower,
            upper=upper,
            m=24,
        )

    metrics_dev = compute_metrics(dev_df)
    metrics_test = compute_metrics(test_df)

    metrics_out = [
        {
            "country": country,
            "model": f"NN_{model_name}",
            "split": "dev",
            **metrics_dev,
        },
        {
            "country": country,
            "model": f"NN_{model_name}",
            "split": "test",
            **metrics_test,
        },
    ]
    pd.DataFrame(metrics_out).to_csv(
        out_metrics_dir / f"{country}_NN_{model_name}_metrics.csv", index=False
    )
    print(f"Saved outputs for {country} {model_name}")

    return metrics_out


if __name__ == "__main__":
    import argparse

    print("Starting neural network experiments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--model", default="gru", choices=["ann", "rnn", "gru", "lstm", "bilstm", "all"]
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")
    models_to_run = (
        ["ann", "rnn", "gru", "lstm", "bilstm"] if args.model == "all" else [args.model]
    )
    print(f"Models to run: {models_to_run}")
    all_metrics = []
    for csv in cfg["cleaned_files"]:
        for m in models_to_run:
            res = run_experiment_for_country(csv, cfg, m, device=args.device)
            all_metrics.extend(res)

    out_metrics_dir = Path(cfg["output_dir"]) / "metrics"
    pd.DataFrame(all_metrics).to_csv(
        out_metrics_dir / "NN_models_summary.csv", index=False
    )
    print("All neural experiments complete.")
