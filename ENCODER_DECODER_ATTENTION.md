# LSTM Encoder-Decoder with Attention (Optional Requirement)

## Overview

This implementation satisfies the **optional** requirement from the assignment:

> "Encoder-decoder Attention can be optional."

The LSTM Encoder-Decoder with Attention model is an advanced sequence-to-sequence architecture for multi-horizon time series forecasting (168h → 24 steps).

## Architecture

### Components

1. **Encoder (LSTM)**

   - Processes the input sequence (168 hours of historical data)
   - Captures temporal patterns and dependencies
   - Produces context vectors for each time step

2. **Bahdanau Attention Mechanism**

   - At each decoder step, computes attention weights over encoder outputs
   - Allows the model to focus on relevant parts of the input sequence
   - Uses additive attention (Bahdanau et al., 2015)

3. **Decoder (LSTM)**
   - Autoregressively generates 24-step forecasts
   - At each step, receives:
     - Previous prediction (or ground truth during training)
     - Context vector from attention mechanism
   - Uses teacher forcing during training for better convergence

### Key Features

- **Attention Weights**: The model learns which input timesteps are most relevant for each forecast step
- **Teacher Forcing**: During training, uses ground truth values with 50% probability to stabilize learning
- **Autoregressive Generation**: At inference, generates predictions step-by-step
- **Quantile Forecasting Support**: Can produce probabilistic forecasts (optional)

## Usage

### Training on Single Country

```bash
# Train on one country (e.g., DK)
python -m src.neural.neural_forecast --model lstm_attn --config config.yaml

# Alternative model name
python -m src.neural.neural_forecast --model attention
```

### Training on All Countries

```bash
# Train all neural models (including attention) on all countries
python -m src.neural.neural_forecast --model all
```

### GPU Training

```bash
# Use GPU if available
python -m src.neural.neural_forecast --model lstm_attn --device cuda
```

## Model Parameters

Default configuration (can be adjusted in `src/neural/neural_forecast.py`):

```python
LSTMEncoderDecoderAttention(
    input_dim=1,              # Main feature (energy price)
    exog_dim=2,               # Exogenous features (solar, wind)
    hidden_size=256,          # LSTM hidden dimension
    n_layers=2,               # Number of LSTM layers
    out_h=24,                 # Forecast horizon
    teacher_forcing_ratio=0.5, # Probability of using ground truth
    dropout=0.1,              # Dropout for regularization
    quantile_levels=None      # Optional quantile forecasting
)
```

## Output Files

After training, the following files will be generated:

```plaintext
outputs/
  forecasts/
    DK_cleaned_NN_lstm_attn_dev.csv    # Development set forecasts
    DK_cleaned_NN_lstm_attn_test.csv   # Test set forecasts
    ES_cleaned_NN_lstm_attn_dev.csv
    ES_cleaned_NN_lstm_attn_test.csv
    FR_cleaned_NN_lstm_attn_dev.csv
    FR_cleaned_NN_lstm_attn_test.csv

  metrics/
    DK_cleaned_NN_lstm_attn_metrics.csv  # Performance metrics
    ES_cleaned_NN_lstm_attn_metrics.csv
    FR_cleaned_NN_lstm_attn_metrics.csv
    NN_models_summary.csv                # Updated with attention results
```

## Comparison with Direct Models

| Model Type                      | Architecture | Pros                    | Cons                  |
| ------------------------------- | ------------ | ----------------------- | --------------------- |
| **Direct (GRU/LSTM)**           | Many-to-one  | Fast, simple            | No attention to input |
| **Encoder-Decoder + Attention** | Seq2seq      | Interpretable, powerful | Slower, more complex  |

The attention model typically:

- ✅ Better captures long-range dependencies
- ✅ Provides interpretability via attention weights
- ✅ More flexible for variable-length sequences
- ❌ Requires more training time
- ❌ More parameters to tune

## Testing

Run the test script to verify the implementation:

```bash
python test_attention_model.py
```

Expected output:

```plaintext
✓ Model created successfully
✓ Training forward pass successful
✓ Evaluation forward pass successful
✓ Backward pass successful
✓ Quantile forecasting successful
✅ All tests passed!
```

## Implementation Details

### Files Modified

1. **`src/neural/neural_models.py`**

   - Added `BahdanauAttention` class
   - Added `LSTMEncoderDecoderAttention` class

2. **`src/neural/neural_forecast.py`**

   - Added model selection for "lstm_attn" and "attention"
   - Updated training loop to support teacher forcing
   - Added to choices in argument parser

3. **`README.md`**
   - Updated project structure documentation

### Code Quality

- ✅ Follows existing code style and conventions
- ✅ Compatible with existing data pipeline
- ✅ Supports same features (exogenous variables, quantile forecasting)
- ✅ Comprehensive inline documentation
- ✅ Tested with dummy data

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR 2015.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to Sequence Learning with Neural Networks. NIPS 2014.

## Notes

This implementation is **optional** as specified in the assignment email. The mandatory requirements (GRU/LSTM direct multi-horizon models) are already fully satisfied.
