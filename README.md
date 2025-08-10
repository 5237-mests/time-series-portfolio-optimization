---

# Week 11 KAIM

## 📌 Project Overview

This project is part of the **10 Academy KAIM Week 11 Challenge**:
**Time Series Forecasting for Portfolio Management Optimization**.
The goal is to apply time series forecasting techniques to financial data (TSLA, BND, SPY) to inform portfolio allocation strategies based on Modern Portfolio Theory (MPT).

---

## 📅 Task 1 — Data Preprocessing & Exploration

**Objective:** Prepare historical asset data for time series modeling.

- Data extraction using Yahoo Finance API (`yfinance`)
- Data cleaning and preparation
- Exploratory Data Analysis (EDA)
- Stationarity testing
- Volatility and risk metrics calculation
- Outlier detection

**Assets Analyzed**

| Asset | Description                               |
| ----- | ----------------------------------------- |
| TSLA  | High-growth, high-volatility stock        |
| BND   | Bond ETF, stability & low risk            |
| SPY   | S\&P 500 ETF, diversified market exposure |

**Completed Steps**

1. **Data Extraction** (2015-07-01 to 2025-07-31)
2. **Cleaning**: Handled missing values via time interpolation
3. **EDA**: Price trends, daily returns, rolling volatility
4. **Outlier Detection**: Top/bottom 2.5% daily returns
5. **Stationarity Testing**:

   - TSLA: Differencing (d=1) required
   - SPY: Differencing (d=1) required
   - BND: Stable, no differencing needed

6. **Risk Metrics**:

   - Sharpe Ratio
   - Value at Risk (VaR, 95%)

**Key Insights**

- TSLA: High potential reward, high volatility
- BND: Stability with minimal daily loss
- SPY: Balanced exposure, moderate risk
- Differencing needed for TSLA & SPY before ARIMA/SARIMA modeling

---

## 📅 Task 2 — Time Series Modeling (ARIMA vs LSTM)

**Objective:** Build and evaluate ARIMA and LSTM forecasting models for TSLA to compare their predictive performance.

**Steps Completed**

1. Implemented **ARIMA** model with rolling 30-day forecast:

   - Differenced TSLA series to achieve stationarity
   - Saved trained ARIMA model (`arima_tsla.joblib`)
   - Generated rolling forecast plot

2. Implemented **LSTM** deep learning model:

   - Normalized series and reshaped for sequence learning
   - Trained with early stopping to prevent overfitting
   - Saved trained LSTM model (`lstm_tsla.h5`)
   - Generated Actual vs Forecast plot

3. Calculated evaluation metrics:

   - **MAE**, **RMSE**, **MAPE** for both models

4. Compared results in a performance table

**Results Summary**

| Model | MAE   | RMSE  | MAPE   |
| ----- | ----- | ----- | ------ |
| ARIMA | 62.98 | 77.94 | 24.11% |
| LSTM  | 10.40 | 14.62 | 3.89%  |

**Key Insights**

- **LSTM significantly outperforms ARIMA** in all error metrics
- ARIMA forecasts capture trend direction but miss magnitude changes
- LSTM better models TSLA’s nonlinear and volatile nature

---
