"""
Online simulation with live ingestion and neural adaptation.
Simulates a stream of data for one country (DK), updates forecasts hourly,
detects anomalies, and adapts the model (fine-tuning) on schedule or drift.
"""

import argparse
import time
import random
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.neural.neural_models import LSTMEncoderDecoderAttention


with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

COUNTRIES = CONFIG["countries"]
SIM_HOURS = 2000
INPUT_WINDOW = 168
HORIZON = 24
QUANTILES = [0.1, 0.5, 0.9]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


Z_WINDOW = 336
Z_MIN_PERIODS = 168
DRIFT_ALPHA = 0.1
DRIFT_HISTORY = 30 * 24
ADAPT_SCHEDULE_HOURS = 6
ADAPT_LR = 1e-4
ADAPT_EPOCHS = 1
ADAPT_HISTORY_DAYS = 14

print(f"Using device: {DEVICE}")


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


class OnlineSimulation:
    def __init__(self, country):
        self.country = country
        self.data_path = Path(f"data/{country}_cleaned.csv")
        self.output_log = Path(f"outputs/{country}_online_updates.csv")

        self.setup_data()
        self.setup_model()
        self.reset_state()

    def setup_data(self):
        print(f"Loading data from {self.data_path}...")
        df = (
            pd.read_csv(self.data_path, parse_dates=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        df["load"] = df["load"].interpolate()

        self.full_df = df
        split_idx = len(df) - SIM_HOURS
        self.history_df = df.iloc[:split_idx].copy()
        self.stream_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        print(f"Initial history: {len(self.history_df)} rows")
        print(f"Stream size: {len(self.stream_df)} rows")

        self.scaler = StandardScaler()
        self.scaler.fit(self.history_df["load"].values.reshape(-1, 1))

    def setup_model(self):
        print("Initializing LSTM model...")

        self.model = LSTMEncoderDecoderAttention(
            input_dim=1,
            exog_dim=0,
            hidden_size=128,
            n_layers=1,
            out_h=HORIZON,
            quantile_levels=QUANTILES,
        ).to(DEVICE)

        self.loss_fn = QuantileLoss(QUANTILES).to(DEVICE)

        print("Training initial model on history (this may take a minute)...")
        self.train_model(self.history_df, epochs=5, lr=1e-3)

    def train_model(self, df, epochs=1, lr=1e-3, freeze_lstm=False):
        self.model.train()

        for name, param in self.model.named_parameters():
            if freeze_lstm and "lstm" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        y = df["load"].values.astype(np.float32)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()

        X_list, Y_list = [], []

        stride = 1 if epochs == 1 else 24
        for i in range(0, len(y_scaled) - INPUT_WINDOW - HORIZON + 1, stride):
            X_list.append(y_scaled[i : i + INPUT_WINDOW])
            Y_list.append(y_scaled[i + INPUT_WINDOW : i + INPUT_WINDOW + HORIZON])

        if not X_list:
            return

        X_t = torch.tensor(np.stack(X_list)).unsqueeze(-1).to(DEVICE)
        Y_t = torch.tensor(np.stack(Y_list)).to(DEVICE)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for ep in range(epochs):
            total_loss = 0
            for bx, by in loader:
                opt.zero_grad()
                pred = self.model(bx)
                loss = self.loss_fn(pred, by)
                loss.backward()
                opt.step()
                total_loss += loss.item()

    def reset_state(self):

        self.current_history = self.history_df.copy()
        self.residuals = []
        self.drift_ewma = 0.0
        self.logs = []
        self.results = []

        self.forecast_buffer = {}

    def get_recent_z_values(self, n=720):

        if not self.residuals:
            return np.array([])
        return np.array(
            [
                r[2]
                for r in self.residuals[-n:]
                if r[2] is not None and not np.isnan(r[2])
            ]
        )

    def compute_metrics_7d(self):

        cutoff = self.current_history["timestamp"].iloc[-1] - pd.Timedelta(days=7)

        recent_df = self.current_history[
            self.current_history["timestamp"] > cutoff
        ].copy()
        if len(recent_df) < INPUT_WINDOW + HORIZON:
            return 0.0, 0.0

        self.model.eval()
        y = recent_df["load"].values.astype(np.float32)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()

        X_list, Y_list = [], []
        for i in range(0, len(y_scaled) - INPUT_WINDOW - HORIZON + 1, 24):
            X_list.append(y_scaled[i : i + INPUT_WINDOW])
            Y_list.append(y_scaled[i + INPUT_WINDOW : i + INPUT_WINDOW + HORIZON])

        if not X_list:
            return 0.0, 0.0

        X_t = torch.tensor(np.stack(X_list)).unsqueeze(-1).to(DEVICE)
        Y_t = np.stack(Y_list)

        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()

        preds_med = preds[:, 1, :]
        preds_lo = preds[:, 0, :]
        preds_hi = preds[:, 2, :]

        Y_true = self.scaler.inverse_transform(Y_t.reshape(-1, 1)).reshape(Y_t.shape)
        Y_hat = self.scaler.inverse_transform(preds_med.reshape(-1, 1)).reshape(
            preds_med.shape
        )
        Y_lo = self.scaler.inverse_transform(preds_lo.reshape(-1, 1)).reshape(
            preds_lo.shape
        )
        Y_hi = self.scaler.inverse_transform(preds_hi.reshape(-1, 1)).reshape(
            preds_hi.shape
        )

        y_full = self.current_history["load"].values
        naive_err = np.mean(np.abs(y_full[24:] - y_full[:-24])) + 1e-8
        mae = np.mean(np.abs(Y_true - Y_hat))
        mase = mae / naive_err

        covered = (Y_true >= Y_lo) & (Y_true <= Y_hi)
        coverage = np.mean(covered) * 100.0

        return mase, coverage

    def run(self):
        print(f"Starting simulation for {SIM_HOURS} hours...")

        print("Warming up z-score stats...")
        warmup_df = self.history_df.iloc[-(Z_WINDOW * 2) :].copy()

        self.residuals = []

        pbar = tqdm(total=len(self.stream_df))

        for i, row in self.stream_df.iterrows():
            ts = row["timestamp"]
            y_true = row["load"]

            if ts.hour == 0 or ts not in self.forecast_buffer:

                last_hist = (
                    self.current_history["load"]
                    .values[-INPUT_WINDOW:]
                    .astype(np.float32)
                )
                if len(last_hist) == INPUT_WINDOW:
                    last_hist_scaled = self.scaler.transform(
                        last_hist.reshape(-1, 1)
                    ).flatten()
                    inp = (
                        torch.tensor(last_hist_scaled)
                        .view(1, INPUT_WINDOW, 1)
                        .to(DEVICE)
                    )

                    self.model.eval()
                    with torch.no_grad():
                        out = self.model(inp).cpu().numpy()

                    out_inv = self.scaler.inverse_transform(out.reshape(-1, 1)).reshape(
                        out.shape
                    )

                    for h in range(HORIZON):
                        fts = ts + pd.Timedelta(hours=h)
                        self.forecast_buffer[fts] = {
                            "yhat": out_inv[0, 1, h],
                            "lo": out_inv[0, 0, h],
                            "hi": out_inv[0, 2, h],
                        }

            fc = self.forecast_buffer.get(ts)
            if fc:
                yhat = fc["yhat"]
                yhat_lo = fc["lo"]
                yhat_hi = fc["hi"]
                resid = y_true - yhat
            else:

                yhat = y_true
                yhat_lo = y_true
                yhat_hi = y_true
                resid = 0.0

            self.residuals.append((ts, resid, 0.0))

            if len(self.residuals) >= Z_MIN_PERIODS:

                recent_resids = np.array([r[1] for r in self.residuals[-Z_WINDOW:]])
                mu = np.mean(recent_resids)
                sigma = np.std(recent_resids) + 1e-8
                z = (resid - mu) / sigma

                self.residuals[-1] = (ts, resid, z)
            else:
                z = 0.0
                self.residuals[-1] = (ts, resid, 0.0)

            abs_z = abs(z)
            self.drift_ewma = DRIFT_ALPHA * abs_z + (1 - DRIFT_ALPHA) * self.drift_ewma

            recent_zs = self.get_recent_z_values(720)
            if len(recent_zs) > 100:
                drift_threshold = np.percentile(np.abs(recent_zs), 95)
            else:
                drift_threshold = 3.0

            drift_triggered = self.drift_ewma > drift_threshold and len(recent_zs) > 100

            strategy = None
            reason = None

            is_scheduled = ts.hour % ADAPT_SCHEDULE_HOURS == 0

            if drift_triggered:
                reason = "drift"
            elif is_scheduled:
                reason = "scheduled"

            if reason:
                strategy = "neural_finetune"
                t0 = time.time()

                mase_pre, cov_pre = self.compute_metrics_7d()

                adapt_df = self.current_history.iloc[
                    -(ADAPT_HISTORY_DAYS * 24) :
                ].copy()
                self.train_model(
                    adapt_df, epochs=ADAPT_EPOCHS, lr=ADAPT_LR, freeze_lstm=True
                )

                mase_post, cov_post = self.compute_metrics_7d()

                dur = time.time() - t0

                self.logs.append(
                    {
                        "timestamp": ts,
                        "strategy": strategy,
                        "reason": reason,
                        "duration_s": round(dur, 2),
                        "mase_pre": round(mase_pre, 3),
                        "mase_post": round(mase_post, 3),
                        "cov_pre": round(cov_pre, 1),
                        "cov_post": round(cov_post, 1),
                        "drift_val": round(self.drift_ewma, 3),
                        "drift_thr": round(drift_threshold, 3),
                    }
                )

                if drift_triggered:
                    self.drift_ewma = 0.0

            self.current_history = pd.concat(
                [self.current_history, row.to_frame().T], ignore_index=True
            )

            self.results.append(
                {
                    "timestamp": ts,
                    "load": y_true,
                    "yhat": yhat,
                    "yhat_lo": yhat_lo,
                    "yhat_hi": yhat_hi,
                    "z_score": z,
                    "drift_ewma": self.drift_ewma,
                    "is_anomaly": abs(z) > 3.0,
                }
            )

            pbar.update(1)

        pbar.close()

        log_df = pd.DataFrame(self.logs)
        self.output_log.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(self.output_log, index=False)

        res_df = pd.DataFrame(self.results)
        res_path = self.output_log.with_name(f"{self.country}_online_results.csv")
        res_df.to_csv(res_path, index=False)

        print(f"\nSimulation complete for {self.country}.")
        print(f"Adaptation logs saved to {self.output_log}")
        print(f"Time series results saved to {res_path}")
        print(log_df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country", type=str, help="Specific country to run (optional)"
    )
    args = parser.parse_args()

    countries_to_run = [args.country] if args.country else COUNTRIES

    for cc in countries_to_run:
        print(f"\n{'='*40}")
        print(f"Running Online Simulation for {cc}")
        print(f"{'='*40}")
        try:
            sim = OnlineSimulation(cc)
            sim.run()
        except Exception as e:
            print(f"Error running simulation for {cc}: {e}")
