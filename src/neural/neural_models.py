"""
Neural forecasters:
 - ANN: Feedforward (flattened window -> 24 outputs)
 - RNN: Vanilla RNN (many-to-one -> 24 outputs)
 - GRU: GRU-based many-to-one
 - LSTM: LSTM-based many-to-one
 - BiLSTM: Bidirectional LSTM many-to-one

All models output a vector of length H (24) directly (direct multi-horizon).
Supports exogenous inputs by concatenating exog features at each time step.
"""

import torch
import torch.nn as nn


class BaseForecaster(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_exog=None):
        raise NotImplementedError


class ANNForecaster(BaseForecaster):
    """Simple Feedforward NN: flatten input window and optional flattened exog."""

    def __init__(
        self,
        in_window=168,
        input_dim=1,
        exog_dim=0,
        hidden_sizes=(256, 128),
        out_h=24,
        dropout=0.1,
        quantile_levels=None,
    ):
        super().__init__()
        self.quantile_levels = quantile_levels
        self.out_h = out_h
        self.n_outputs = len(quantile_levels) if quantile_levels else 1
        self.in_features = in_window * input_dim + (
            in_window * exog_dim if exog_dim and False else exog_dim
        )

        flatten_size = (
            in_window * input_dim + in_window * exog_dim
            if exog_dim
            else in_window * input_dim
        )

        layers = []
        prev = flatten_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_h * self.n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x, x_exog=None):

        B, T, D = x.shape
        if x_exog is not None:
            xe = x_exog.reshape(B, T * x_exog.shape[2])
            xin = torch.cat([x.reshape(B, T * D), xe], dim=1)
        else:
            xin = x.reshape(B, T * D)
        out = self.net(xin)
        if self.quantile_levels:
            out = out.view(out.size(0), self.n_outputs, self.out_h)
        return out


class RNNForecaster(BaseForecaster):
    """Simple RNN-based direct forecaster"""

    def __init__(
        self,
        input_dim=1,
        exog_dim=0,
        hidden_size=128,
        n_layers=1,
        out_h=24,
        dropout=0.1,
        quantile_levels=None,
    ):
        super().__init__()
        self.input_dim = input_dim + exog_dim
        self.out_h = out_h
        self.quantile_levels = quantile_levels
        self.n_outputs = len(quantile_levels) if quantile_levels else 1
        self.rnn = nn.RNN(
            self.input_dim, hidden_size, n_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, out_h * self.n_outputs)

    def forward(self, x, x_exog=None):

        if x_exog is not None:
            xin = torch.cat([x, x_exog], dim=2)
        else:
            xin = x

        out, hn = self.rnn(xin)
        last = hn[-1]
        out = self.fc(last)
        if self.quantile_levels:
            out = out.view(out.size(0), self.n_outputs, self.out_h)
        return out


class GRUForecaster(BaseForecaster):
    """GRU many-to-one direct multi-horizon"""

    def __init__(
        self,
        input_dim=1,
        exog_dim=0,
        hidden_size=128,
        n_layers=1,
        out_h=24,
        dropout=0.1,
        bidirectional=False,
        quantile_levels=None,
    ):
        super().__init__()
        self.input_dim = input_dim + exog_dim
        self.out_h = out_h
        self.quantile_levels = quantile_levels
        self.n_outputs = len(quantile_levels) if quantile_levels else 1
        self.gru = nn.GRU(
            self.input_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, max(64, self.hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.hidden_size // 2), out_h * self.n_outputs),
        )

    def forward(self, x, x_exog=None):
        if x_exog is not None:
            xin = torch.cat([x, x_exog], dim=2)
        else:
            xin = x
        out, hn = self.gru(xin)

        if isinstance(hn, torch.Tensor):
            last = hn.view(hn.size(0), hn.size(1), hn.size(2))[-1]
        else:
            last = out[:, -1, :]

        out = self.fc(last)
        if self.quantile_levels:
            out = out.view(out.size(0), self.n_outputs, self.out_h)
        return out


class LSTMForecaster(BaseForecaster):
    """LSTM many-to-one multi-horizon direct output"""

    def __init__(
        self,
        input_dim=1,
        exog_dim=0,
        hidden_size=128,
        n_layers=1,
        out_h=24,
        dropout=0.1,
        bidirectional=False,
        quantile_levels=None,
    ):
        super().__init__()
        self.input_dim = input_dim + exog_dim
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1
        self.out_h = out_h
        self.quantile_levels = quantile_levels
        self.n_outputs = len(quantile_levels) if quantile_levels else 1
        self.lstm = nn.LSTM(
            self.input_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.hidden_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, max(64, self.hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.hidden_size // 2), out_h * self.n_outputs),
        )

    def forward(self, x, x_exog=None):
        if x_exog is not None:
            xin = torch.cat([x, x_exog], dim=2)
        else:
            xin = x
        out, (hn, cn) = self.lstm(xin)

        hn = hn.view(self.n_layers, self.num_directions, hn.size(1), hn.size(2))
        last_layer = hn[-1]
        last = last_layer.transpose(0, 1).reshape(last_layer.size(1), -1)
        out = self.fc(last)
        if self.quantile_levels:
            out = out.view(out.size(0), self.n_outputs, self.out_h)
        return out


class BiLSTMForecaster(LSTMForecaster):
    """Convenience subclass for bidirectional LSTM"""

    def __init__(
        self,
        input_dim=1,
        exog_dim=0,
        hidden_size=128,
        n_layers=1,
        out_h=24,
        dropout=0.1,
        quantile_levels=None,
    ):
        super().__init__(
            input_dim=input_dim,
            exog_dim=exog_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            out_h=out_h,
            dropout=dropout,
            bidirectional=True,
            quantile_levels=quantile_levels,
        )
