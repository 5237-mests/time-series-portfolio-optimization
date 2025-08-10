"""
LSTM pipeline for TSLA forecasting.

Requirements:
  - pandas, numpy, matplotlib, scikit-learn, tensorflow, joblib, yfinance (optional)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def ensure_data(path_processed="../data/processed/tsla.csv", path_raw="../data/raw/tsla.csv", start="2015-07-01", end="2025-07-31"):
    if os.path.exists(path_processed):
        df = pd.read_csv(path_processed, parse_dates=["Date"], index_col="Date")
        return df
    if os.path.exists(path_raw):
        df = pd.read_csv(path_raw, parse_dates=["Date"], index_col="Date")
    else:
        try:
            import yfinance as yf
            df = yf.download("TSLA", start=start, end=end)
            df.index = pd.to_datetime(df.index)
            os.makedirs(os.path.dirname(path_raw), exist_ok=True)
            df.to_csv(path_raw)
        except Exception as e:
            raise RuntimeError("No data found and unable to fetch. Place CSV at data/processed/tsla.csv") from e

    # feature engineering
    df = df.sort_index()
    df["Close"] = df["Close"].astype(float)
    df["Daily Return"] = df["Close"].pct_change()
    df["Rolling Vol"] = df["Daily Return"].rolling(window=30).std()
    os.makedirs(os.path.dirname(path_processed), exist_ok=True)
    df.to_csv(path_processed)
    return df

def train_test_split_series(series, train_end="2023-12-31"):
    train = series[:train_end].dropna()
    test = series[train_end:].dropna()
    return train, test

def create_sequences(values, lookback=60):
    """
    values: 1D numpy array (scaled)
    returns X (samples, lookback, 1) and y (samples,)
    """
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def build_lstm(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def plot_results(train, test, y_pred, save_path=None):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(test.index, y_pred, linestyle="--", label="LSTM Forecast")
    plt.title("TSLA: Actual vs LSTM Forecast")
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def run_lstm_pipeline(processed_csv="../../data/processed/tsla.csv",
                      model_out="models/lstm_tsla.h5",
                      scaler_out="models/lstm_scaler.gz",
                      plot_out="../../reports/lstm_tsla_forecast.png",
                      lookback=60,
                      epochs=50,
                      batch_size=32,
                      train_end="2023-12-31",
                      val_split=0.1):
    # Load data
    df = ensure_data(path_processed=processed_csv)
    series = df["Close"].sort_index()
    train, test = train_test_split_series(series, train_end=train_end)

    print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)})")
    print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)})")

    # Scale
    scaler = MinMaxScaler(feature_range=(0,1))
    train_vals = train.values.reshape(-1,1)
    scaler.fit(train_vals)  # fit only on train
    train_scaled = scaler.transform(train_vals).flatten()
    # For test we will transform using same scaler
    test_vals = test.values.reshape(-1,1)
    test_scaled = scaler.transform(test_vals).flatten()

    # Create sequences from the concatenation of train+test for rolling predictions later OR separate approach:
    X_train, y_train = create_sequences(train_scaled, lookback=lookback)
    # For test, we'll perform rolling predictions: use last `lookback` days from train + progressive steps
    # Simpler approach: build sequences from concatenated series so that test sequences are aligned with test set
    total_scaled = np.concatenate([train_scaled, test_scaled])
    X_all, y_all = create_sequences(total_scaled, lookback=lookback)

    # X_all corresponds to dates starting at (start_date + lookback days)
    # Figure index mapping
    all_index = series.index[lookback:]  # index aligned to y_all

    # Determine split index
    split_point = len(train_scaled) - lookback  # number of samples in X_all that belong to train
    X_train = X_all[:split_point]
    y_train = y_all[:split_point]
    X_test = X_all[split_point:]
    y_test = y_all[split_point:]
    test_index = all_index[split_point:]

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Build model
    model = build_lstm(input_shape=(lookback, 1), units=50, dropout=0.2)
    model.summary()

    # Callbacks
    early = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

    # Fit
    history = model.fit(
        X_train, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=1
    )

    # Predict on X_test
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform predictions and y_test
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    # Build pandas Series aligned to test_index
    y_pred_series = pd.Series(data=y_pred, index=test_index)
    y_true_series = pd.Series(data=y_true, index=test_index)

    # Evaluate
    metrics = evaluate_forecast(y_true_series.values, y_pred_series.values)
    print("Evaluation:")
    for k,v in metrics.items():
        if k=="MAPE":
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")

    # Plot
    plot_results(train, test, y_pred_series, save_path=plot_out)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    model.save(model_out)
    joblib.dump(scaler, scaler_out)
    print(f"Saved LSTM model to {model_out} and scaler to {scaler_out}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred_series,
        "y_true": y_true_series,
        "metrics": metrics,
        "history": history
    }

if __name__ == "__main__":
    res = run_lstm_pipeline()
