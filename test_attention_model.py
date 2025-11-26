"""
Quick test script to verify the LSTM Encoder-Decoder with Attention model works correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.neural.neural_models import LSTMEncoderDecoderAttention


def test_attention_model():
    """Test the attention model with dummy data"""
    print("Testing LSTM Encoder-Decoder with Attention...")

    # Model parameters
    batch_size = 8
    input_window = 168
    horizon = 24
    input_dim = 1
    exog_dim = 2
    hidden_size = 64

    # Create model
    model = LSTMEncoderDecoderAttention(
        input_dim=input_dim,
        exog_dim=exog_dim,
        hidden_size=hidden_size,
        n_layers=2,
        out_h=horizon,
        teacher_forcing_ratio=0.5,
        quantile_levels=None,
    )

    print(f"✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    x = torch.randn(batch_size, input_window, input_dim)
    x_exog = torch.randn(batch_size, input_window, exog_dim)
    y_true = torch.randn(batch_size, horizon)

    # Test forward pass (training mode with teacher forcing)
    model.train()
    output_train = model(x, x_exog, y_true)
    print(f"✓ Training forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Exog shape: {x_exog.shape}")
    print(f"  Output shape: {output_train.shape}")
    print(f"  Expected: ({batch_size}, {horizon})")

    assert output_train.shape == (
        batch_size,
        horizon,
    ), f"Shape mismatch: {output_train.shape}"

    # Test forward pass (inference mode without teacher forcing)
    model.eval()
    with torch.no_grad():
        output_eval = model(x, x_exog)
    print(f"✓ Evaluation forward pass successful")
    print(f"  Output shape: {output_eval.shape}")

    # Test backward pass
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    optimizer.zero_grad()
    output = model(x, x_exog, y_true)
    loss = loss_fn(output, y_true)
    loss.backward()
    optimizer.step()

    print(f"✓ Backward pass successful")
    print(f"  Loss: {loss.item():.4f}")

    # Test with quantile forecasting
    print("\nTesting with quantile forecasting...")
    model_quantile = LSTMEncoderDecoderAttention(
        input_dim=input_dim,
        exog_dim=exog_dim,
        hidden_size=hidden_size,
        n_layers=2,
        out_h=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    model_quantile.eval()
    with torch.no_grad():
        output_quantile = model_quantile(x, x_exog)
    print(f"✓ Quantile forecasting successful")
    print(f"  Output shape: {output_quantile.shape}")
    print(f"  Expected: ({batch_size}, 3, {horizon})")

    assert output_quantile.shape == (
        batch_size,
        3,
        horizon,
    ), f"Shape mismatch: {output_quantile.shape}"

    print("\n" + "=" * 60)
    print("✅ All tests passed! Encoder-Decoder with Attention is ready.")
    print("=" * 60)
    print("\nTo train the model on your data, run:")
    print("  python -m src.neural.neural_forecast --model lstm_attn")
    print("  python -m src.neural.neural_forecast --model attention")


if __name__ == "__main__":
    test_attention_model()
