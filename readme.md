# Energy Demand Forecasting using LSTM & Attention

## Project Overview

This project implements an end-to-end multi-horizon time-series forecasting system using Long Short-Term Memory (LSTM) networks and an attention mechanism.

The objective is to forecast the next 24 hours of energy demand using the previous 168 hours of historical observations. The project uses a completely synthetic energy-demand dataset generated in Python, incorporating realistic temporal patterns such as daily seasonality, weekly seasonality, an upward trend, weekend demand reductions, and random noise.

Three forecasting approaches are evaluated:

1. Seasonal Naive Forecasting (Baseline)
2. LSTM Encoder–Decoder
3. LSTM Encoder–Decoder with Attention

The primary objective is to determine whether deep learning, particularly attention-based sequence modelling, can outperform a traditional seasonal forecasting approach.

## Dataset Description

The dataset is generated entirely within the notebook without downloading external files or accessing APIs.

**Dataset characteristics:**

- Duration: 2 years
- Frequency: Hourly
- Total observations: 17,472
- Daily seasonality: 24 hours
- Weekly seasonality: 168 hours
- Trend: Gradual upward demand drift
- Weekend effect: Reduced demand on Saturdays and Sundays
- Noise: Gaussian random noise

The generated demand series combines these components:

- Long-term upward trend
- Daily demand fluctuations
- Weekly demand patterns
- Weekend demand reduction
- Random variations

## Project Configuration

| Parameter | Configuration |
|---|---|
| Look-back window | 168 hours |
| Forecast horizon | 24 hours |
| Training split | 70% |
| Validation split | 15% |
| Testing split | 15% |
| Forecasting frequency | Hourly |
| Framework | TensorFlow / Keras |
| Architecture | LSTM Encoder–Decoder |
| Attention | Keras Temporal Attention |
| Optimizer | Adam |
| Loss function | Mean Squared Error |
| Random seed | 42 |

## Methodology

### 1. Synthetic Data Generation

A two-year hourly energy-demand series is generated using NumPy.

The dataset incorporates multiple seasonal components and random noise to simulate realistic demand behaviour.

### 2. Sliding Window Transformation

The time series is converted into supervised learning samples.

Each training example consists of:

- Input (X): Previous 168 hours of energy demand.
- Target (y): Next 24 hours of energy demand.

This transforms the forecasting task into a multi-output regression problem.

### 3. Chronological Data Splitting

The dataset is divided into training, validation, and testing periods without random shuffling.

This ensures that future observations are never used to train models that predict the past.

Normalization is performed using only the training-period mean and standard deviation to prevent data leakage.

### 4. Seasonal Naive Baseline

Before implementing deep learning, a seasonal-naive forecasting model is established.

The baseline predicts future demand using the corresponding observations from the previous week.

Since the dataset contains strong weekly seasonality, this provides a meaningful benchmark for evaluating the neural networks.

### 5. LSTM Encoder–Decoder

An LSTM encoder processes the previous 168 hours and extracts a compact representation of the historical demand pattern.

The decoder uses this representation to generate predictions for the next 24 hours.

The architecture captures temporal dependencies and nonlinear relationships within the sequence.

### 6. LSTM with Attention

The attention-based architecture extends the encoder–decoder model.

Instead of relying exclusively on the final encoder state, the attention mechanism accesses the complete sequence of encoder hidden states.

Each decoder time step calculates attention weights over historical encoder outputs and generates a context vector.

The context vector is combined with the decoder output before producing the final forecast.

This allows the model to focus on different historical time steps when forecasting different future hours.

## Model Evaluation

Three evaluation metrics are used:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error; measures average absolute prediction error. |
| RMSE | Root Mean Squared Error; penalizes larger forecasting errors more heavily. |
| MAPE | Mean Absolute Percentage Error; expresses prediction error as a percentage. |

The notebook automatically generates a comparison table containing the test-set performance of all three approaches.

**Actual experimental results:** Refer to the generated metric comparison table in the notebook. Metrics are intentionally not hardcoded because they are produced during model execution.

## Visualizations

The notebook generates:

- Synthetic energy-demand time-series visualization.
- LSTM training and validation loss curves.
- LSTM with attention training and validation loss curves.
- Actual versus predicted 24-hour demand comparison.
- Model performance comparison using MAE, RMSE, and MAPE.

## Challenges and Key Learnings

### Challenges Faced

1. **Time-Series Windowing:** Designing sliding windows that correctly map 168 hours of historical observations to the following 24 hours.

2. **Data Leakage Prevention:** Ensuring that normalization statistics are calculated exclusively from the training period.

3. **Chronological Splitting:** Maintaining temporal order during dataset partitioning instead of applying conventional random train-test splitting.

4. **Multi-Horizon Forecasting:** Predicting multiple future time steps simultaneously rather than predicting only the next observation.

5. **Attention Integration:** Preserving encoder hidden states and integrating attention context into the decoder architecture.

6. **Baseline Comparison:** Understanding why a traditional seasonal-naive model can remain competitive when the underlying time series contains strong recurring patterns.

### Key Learnings

- Understanding LSTM encoder–decoder architectures.
- Implementing multi-horizon forecasting.
- Applying temporal attention mechanisms.
- Understanding seasonal-naive forecasting.
- Preventing data leakage in time-series modelling.
- Evaluating forecasting models using multiple regression metrics.
- Comparing model complexity against actual performance improvements.

## Teacher Forcing vs Autoregressive Inference

This implementation uses a repeated encoder context as the decoder input rather than feeding previous ground-truth target values into the decoder.

Therefore, the model does not use traditional teacher forcing.

The decoder generates the complete 24-hour forecast in one forward pass. This avoids autoregressive error accumulation during inference and maintains consistency between training and prediction.

## Conclusion

This capstone demonstrates an end-to-end multi-horizon forecasting pipeline using recurrent neural networks and attention mechanisms.

The central objective is not simply to achieve the lowest forecasting error but to understand whether additional architectural complexity provides measurable improvements over a strong seasonal baseline.

The comparison between the seasonal-naive model, LSTM encoder–decoder, and attention-enhanced LSTM provides practical insight into the strengths and limitations of deep learning for structured time-series forecasting.

An attention mechanism should only be considered beneficial when its performance improvements justify the additional complexity.

## Repository Structure

```text
Day22/
│
├── day22_capstone.ipynb
└── README.md
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras
- Jupyter Notebook

## How to Run

Clone the repository:

```bash
git clone <your-repository-url>
cd Day22
```

Install dependencies:

```bash
pip install numpy pandas matplotlib tensorflow jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `day22_capstone.ipynb` and execute all cells sequentially.

**Note:** The dataset is generated automatically. No external dataset or internet connection is required for data acquisition.

---

**Day 22 | Sequence Models | LSTM + Attention | Multi-Horizon Time-Series Forecasting**
