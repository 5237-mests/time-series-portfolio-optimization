"""
ARIMA pipeline for TSLA forecasting.
Includes simple grid search for best (p,d,q) based on AIC.
Also includes rolling ARIMA for short-term dynamic forecasts.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def ensure_data(path_raw="../data/raw/tsla.csv", path_processed="../data/processed/tsla.csv",
                start="2015-07-01", end="2025-07-31"):
    if os.path.exists(path_processed):
        df = pd.read_csv(path_processed, parse_dates=["Date"], index_col="Date")
        return df
    if os.path.exists(path_raw):
        df = pd.read_csv(path_raw)
    else:
        try:
            import yfinance as yf
            df = yf.download("TSLA", start=start, end=end)
            df.index = pd.to_datetime(df.index)
            os.makedirs(os.path.dirname(path_raw), exist_ok=True)
            df.to_csv(path_raw)
        except Exception as e:
            raise RuntimeError("No data found and unable to fetch. Place CSV at data/processed/tsla.csv") from e

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


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_results(train, test, forecast, title, save_path=None):
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train.values, label="Train")
    plt.plot(test.index, test.values, label="Test")
    plt.plot(forecast.index, forecast.values, linestyle="--", label="ARIMA Forecast")
    plt.title(title)
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def grid_search_arima(train, p_values=range(0,4), d_values=range(0,3), q_values=range(0,4)):
    best_aic = float("inf")
    best_order = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train, order=(p,d,q))
                    model_fit = model.fit()
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p,d,q)
                except:
                    continue
    return best_order, best_aic


def rolling_arima_forecast(series, order, train_size=0.8, horizon=30):
    """Perform rolling ARIMA forecast with short-term horizon."""
    # Ensure Series is 1D
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()

    series = pd.Series(series, dtype="float64")

    n_total = len(series)
    n_train = int(train_size * n_total)

    # Keep as Series with float dtype
    history = series.iloc[:n_train].copy()
    test = series.iloc[n_train:].copy()
    predictions = []
    test_index = test.index

    for start in range(0, len(test), horizon):
        # train_chunk = history.dropna()
        train_chunk = pd.Series(history.values, index=history.index, dtype="float64").dropna()
        model = ARIMA(train_chunk, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=min(horizon, len(test) - start))
        predictions.extend(forecast.tolist())

        # Append actuals to history
        history = pd.concat([history, test.iloc[start:start+horizon]])

    pred_series = pd.Series(predictions, index=test_index, dtype="float64")
    return pred_series, test


def run_arima_pipeline(processed_csv="../data/processed/tsla.csv",
                       model_out="../models/arima_tsla.joblib",
                       plot_out="../reports/arima_tsla_forecast.png",
                       train_end="2023-12-31", order=None):

    df = ensure_data(path_processed=processed_csv)
    series = df["Close"].sort_index()

    train, test = train_test_split_series(series, train_end=train_end)
    print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)})")
    print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)})")

    if order is None:
        print("Searching best ARIMA parameters (p,d,q)...")
        order, best_aic = grid_search_arima(train)
        print(f"Best order: {order} with AIC={best_aic:.2f}")

    # --- Baseline full-period forecast ---
    print(f"\nFitting ARIMA{order} on training set...")
    model = ARIMA(train, order=order)
    model_fit = model.fit()
    print(model_fit.summary())

    forecast_res = model_fit.get_forecast(steps=len(test))
    forecast = forecast_res.predicted_mean
    forecast.index = test.index

    metrics = evaluate_forecast(test.values, forecast.values)
    print("\nBaseline Full Forecast Evaluation:")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}" if k != "MAPE" else f"{k}: {v:.2f}%")

    plot_results(train, test, forecast, "TSLA: Baseline ARIMA Forecast", save_path=plot_out)

    # Save baseline model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model_fit, model_out)
    print(f"Saved baseline ARIMA model to {model_out}")

    # --- Rolling short-term forecast ---
    print("\nRunning rolling ARIMA forecast...")
    rolling_preds, rolling_test = rolling_arima_forecast(series, order=order, horizon=30)
    rolling_metrics = evaluate_forecast(rolling_test.values, rolling_preds.values)
    print("\nRolling Forecast Evaluation:")
    for k,v in rolling_metrics.items():
        print(f"{k}: {v:.4f}" if k != "MAPE" else f"{k}: {v:.2f}%")

    plot_results(series[:rolling_test.index[0]], rolling_test, rolling_preds,
                 "TSLA: Rolling 30-Day ARIMA Forecast",
                 save_path="../reports/arima_tsla_rolling.png")

    return {
        "baseline": {"model": model_fit, "forecast": forecast, "y_true": test, "metrics": metrics},
        "rolling": {"forecast": rolling_preds, "y_true": rolling_test, "metrics": rolling_metrics}
    }


if __name__ == "__main__":
    run_arima_pipeline()







# """
# ARIMA pipeline for TSLA forecasting.
# Includes simple grid search for best (p,d,q) based on AIC.
# """

# import os
# import warnings
# warnings.filterwarnings("ignore")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import joblib

# def ensure_data(path_raw="../data/raw/tsla.csv", path_processed="../data/processed/tsla.csv",
#                 start="2015-07-01", end="2025-07-31"):
#     if os.path.exists(path_processed):
#         df = pd.read_csv(path_processed, parse_dates=["Date"], index_col="Date")
#         return df
#     # else try to load raw or fetch
#     if os.path.exists(path_raw):
#         df = pd.read_csv(path_raw, parse_dates=["Date"], index_col="Date")
#     else:
#         try:
#             import yfinance as yf
#             df = yf.download("TSLA", start=start, end=end)
#             df.index = pd.to_datetime(df.index)
#             os.makedirs(os.path.dirname(path_raw), exist_ok=True)
#             df.to_csv(path_raw)
#         except Exception as e:
#             raise RuntimeError("No data found and unable to fetch. Place CSV at data/processed/tsla.csv") from e

#     df["Close"] = df["Close"].astype(float)
#     df["Daily Return"] = df["Close"].pct_change()
#     df["Rolling Vol"] = df["Daily Return"].rolling(window=30).std()
#     os.makedirs(os.path.dirname(path_processed), exist_ok=True)
#     df.to_csv(path_processed)
#     return df


# def train_test_split_series(series, train_end="2023-12-31"):
#     train = series[:train_end].dropna()
#     test = series[train_end:].dropna()
#     return train, test


# def evaluate_forecast(y_true, y_pred):
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# def plot_results(train, test, forecast, save_path=None):
#     plt.figure(figsize=(12,6))
#     plt.plot(train.index, train.values, label="Train")
#     plt.plot(test.index, test.values, label="Test")
#     plt.plot(forecast.index, forecast.values, linestyle="--", label="ARIMA Forecast")
#     plt.title("TSLA: Actual vs ARIMA Forecast")
#     plt.legend()
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, bbox_inches="tight", dpi=150)
#     plt.show()


# def grid_search_arima(train, p_values=range(0,4), d_values=range(0,3), q_values=range(0,4)):
#     """Simple grid search for best (p,d,q) using AIC."""
#     best_aic = float("inf")
#     best_order = None
#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 try:
#                     model = ARIMA(train, order=(p,d,q))
#                     model_fit = model.fit()
#                     if model_fit.aic < best_aic:
#                         best_aic = model_fit.aic
#                         best_order = (p,d,q)
#                 except:
#                     continue
#     return best_order, best_aic


# def run_arima_pipeline(processed_csv="../data/processed/tsla.csv", model_out="../models/arima_tsla.joblib",             plot_out="../reports/arima_tsla_forecast.png",               train_end="2023-12-31", order=None):
#     # Load data
#     df = ensure_data(path_processed=processed_csv)
#     series = df["Close"].sort_index()

#     train, test = train_test_split_series(series, train_end=train_end)
#     print(f"Train: {train.index.min().date()} - {train.index.max().date()} ({len(train)})")
#     print(f"Test:  {test.index.min().date()} - {test.index.max().date()} ({len(test)})")

#     # If no order given, run grid search
#     if order is None:
#         print("Searching best ARIMA parameters (p,d,q)...")
#         order, best_aic = grid_search_arima(train)
#         print(f"Best order: {order} with AIC={best_aic:.2f}")

#     # Fit ARIMA
#     print(f"Fitting ARIMA{order}...")
#     model = ARIMA(train, order=order)
#     model_fit = model.fit()
#     print(model_fit.summary())

#     # Forecast for the test period
#     forecast_res = model_fit.get_forecast(steps=len(test))
#     forecast = forecast_res.predicted_mean
#     forecast.index = test.index

#     # Evaluate
#     metrics = evaluate_forecast(test.values, forecast.values)
#     print("Evaluation:")
#     for k,v in metrics.items():
#         if k=="MAPE":
#             print(f"{k}: {v:.2f}%")
#         else:
#             print(f"{k}: {v:.4f}")

#     # Plot
#     plot_results(train, test, forecast, save_path=plot_out)

#     # Save model
#     os.makedirs(os.path.dirname(model_out), exist_ok=True)
#     joblib.dump(model_fit, model_out)
#     print(f"Saved ARIMA model to {model_out}")

#     return {
#         "model": model_fit,
#         "forecast": forecast,
#         "y_true": test,
#         "metrics": metrics
#     }


# if __name__ == "__main__":
#     run_arima_pipeline()
