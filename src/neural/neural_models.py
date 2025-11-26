"""
Neural forecasters:
 - ANN: Feedforward (flattened window -> 24 outputs)
 - RNN: Vanilla RNN (many-to-one -> 24 outputs)
 - GRU: GRU-based many-to-one
 - LSTM: LSTM-based many-to-one
 - BiLSTM: Bidirectional LSTM many-to-one
 - LSTMEncoderDecoderAttention: Seq2seq with attention (encoder-decoder architecture)

All models output a vector of length H (24) directly (direct multi-horizon).
Supports exogenous inputs by concatenating exog features at each time step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism for sequence-to-sequence models"""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: (B, hidden_size) - current decoder state
        encoder_outputs: (B, T, hidden_size) - all encoder outputs

        Returns:
        context: (B, hidden_size) - weighted sum of encoder outputs
        attention_weights: (B, T) - attention distribution
        """
        # Expand decoder hidden to match encoder outputs time dimension
        # (B, hidden_size) -> (B, T, hidden_size)
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(
            1, encoder_outputs.size(1), 1
        )

        # Compute alignment scores
        # (B, T, hidden_size)
        energy = torch.tanh(
            self.W_a(decoder_hidden_expanded) + self.U_a(encoder_outputs)
        )

        # (B, T, 1) -> (B, T)
        scores = self.v_a(energy).squeeze(2)

        # Normalize to get attention weights
        attention_weights = F.softmax(scores, dim=1)

        # Compute context vector as weighted sum
        # (B, T, 1) x (B, T, hidden_size) -> (B, hidden_size)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class LSTMEncoderDecoderAttention(BaseForecaster):
    """
    LSTM Encoder-Decoder with Bahdanau Attention for multi-horizon forecasting.

    Architecture:
    - Encoder: LSTM processes input sequence (168 hours)
    - Decoder: LSTM autoregressively generates forecast (24 steps)
    - Attention: At each decoder step, attends to encoder outputs

    This is the optional advanced model mentioned in the assignment.
    """

    def __init__(
        self,
        input_dim=1,
        exog_dim=0,
        hidden_size=128,
        n_layers=1,
        out_h=24,
        dropout=0.1,
        teacher_forcing_ratio=0.5,
        quantile_levels=None,
    ):
        super().__init__()
        self.input_dim = input_dim + exog_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_h = out_h
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.quantile_levels = quantile_levels
        self.n_outputs = len(quantile_levels) if quantile_levels else 1

        # Encoder: processes input sequence
        self.encoder = nn.LSTM(
            self.input_dim,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Attention mechanism
        self.attention = BahdanauAttention(hidden_size)

        # Decoder: generates forecast autoregressively
        # Input: previous prediction + context from attention
        self.decoder = nn.LSTM(
            1 + hidden_size,  # previous value + context vector
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_size, self.n_outputs)

        # Learnable initial decoder input
        self.decoder_input_init = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x, x_exog=None, y_true=None):
        """
        x: (B, T_in, 1) - input sequence
        x_exog: (B, T_in, exog_dim) - optional exogenous features
        y_true: (B, T_out) - ground truth for teacher forcing during training

        Returns:
        predictions: (B, T_out) or (B, n_outputs, T_out) if quantile forecasting
        """
        if x_exog is not None:
            encoder_input = torch.cat([x, x_exog], dim=2)
        else:
            encoder_input = x

        batch_size = x.size(0)
        device = x.device

        # Encode input sequence
        encoder_outputs, (h_n, c_n) = self.encoder(encoder_input)
        # encoder_outputs: (B, T_in, hidden_size)
        # h_n: (n_layers, B, hidden_size)
        # c_n: (n_layers, B, hidden_size)

        # Initialize decoder hidden states with encoder final states
        decoder_h = h_n
        decoder_c = c_n

        # Prepare for decoding
        predictions = []

        # Initial decoder input (last value from input)
        decoder_input = x[:, -1, :1]  # (B, 1) - last value

        # Autoregressive decoding
        for t in range(self.out_h):
            # Get current decoder hidden state for attention
            # Use last layer's hidden state
            current_h = decoder_h[-1]  # (B, hidden_size)

            # Apply attention
            context, attn_weights = self.attention(current_h, encoder_outputs)
            # context: (B, hidden_size)

            # Concatenate previous prediction with context
            # decoder_input: (B, 1), context: (B, hidden_size)
            # Combine into (B, 1 + hidden_size)
            decoder_input_combined = torch.cat([decoder_input, context], dim=1)
            # (B, 1 + hidden_size) -> add sequence dimension -> (B, 1, 1 + hidden_size)
            decoder_input_combined = decoder_input_combined.unsqueeze(1)

            # Decoder step
            decoder_output, (decoder_h, decoder_c) = self.decoder(
                decoder_input_combined, (decoder_h, decoder_c)
            )
            # decoder_output: (B, 1, hidden_size)

            # Generate prediction
            pred = self.fc_out(decoder_output.squeeze(1))  # (B, n_outputs)
            predictions.append(pred)

            # Prepare next decoder input (teacher forcing)
            if (
                self.training
                and y_true is not None
                and torch.rand(1).item() < self.teacher_forcing_ratio
            ):
                # Use ground truth
                decoder_input = y_true[:, t : t + 1]  # (B, 1)
            else:
                # Use model's own prediction
                if self.n_outputs > 1:
                    # Use median prediction for next input
                    median_idx = len(self.quantile_levels) // 2
                    decoder_input = pred[:, median_idx : median_idx + 1]  # (B, 1)
                else:
                    decoder_input = pred  # (B, 1) already if n_outputs=1
                    if decoder_input.dim() == 1:
                        decoder_input = decoder_input.unsqueeze(1)

        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, T_out, n_outputs)

        if self.quantile_levels:
            predictions = predictions.transpose(1, 2)  # (B, n_outputs, T_out)
        else:
            predictions = predictions.squeeze(2)  # (B, T_out)

        return predictions
