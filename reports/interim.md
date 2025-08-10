.

---

# **Week 11 KAIM Interim Report — Task 1**

**Project:** Time Series Forecasting for Portfolio Management Optimization
**Analyst:** Mesfin Mulugeta
**Date:** 2025, Augest 10

---

## **1. Objective**

The purpose of Task 1 was to **extract, clean, and explore historical financial data** for three key assets — **TSLA, BND, and SPY** — in preparation for time series forecasting and portfolio optimization. The analysis focused on understanding trends, volatility, stationarity, and risk-return characteristics.

---

## **2. Data Extraction**

- Data Source: **Yahoo Finance** via the `yfinance` Python API
- Period: **2015-07-01 to 2025-07-31**
- Assets:

  - **TSLA** — High-growth, high-volatility stock
  - **BND** — Bond ETF (stability, low risk)
  - **SPY** — S\&P 500 ETF (diversification, moderate risk)

---

## **3. Data Cleaning & Preparation**

- Checked and confirmed correct **data types**
- Missing values handled via **time-based interpolation**
- Calculated **daily returns** (`pct_change()`) for all assets
- Added **30-day rolling mean and rolling standard deviation** to measure volatility
- Normalization not applied at this stage (ARIMA doesn’t require it; will be applied for LSTM)

---

## **4. Exploratory Data Analysis (EDA)**

### **4.1 Price Trends**

- **TSLA** shows a significant long-term upward trend with sharp fluctuations.
- **SPY** exhibits steady growth with moderate dips (notably during market shocks).
- **BND** remains stable with minimal directional changes.

### **4.2 Daily Returns**

- **TSLA** has the widest daily return range (indicative of higher volatility).
- **BND** daily returns are tightly clustered around zero.
- **SPY** lies between TSLA and BND in volatility.

### **4.3 Volatility**

- Rolling 30-day standard deviation confirms:

  - TSLA experiences frequent volatility spikes.
  - BND is consistently low-volatility.
  - SPY shows market-driven volatility patterns (e.g., COVID-19 crash).

### **4.4 Outlier Detection**

- Outliers identified as top/bottom **2.5%** daily returns for each asset.
- TSLA’s largest daily gain exceeded **+20%**, largest loss exceeded **-10%**.
- SPY and BND exhibited far smaller extreme values.

---

## **5. Stationarity Testing**

- **ADF Test Results** (p-value threshold: 0.05):

  - TSLA: Non-stationary → Differencing applied → Stationary
  - SPY: Non-stationary → Differencing applied → Stationary
  - BND: Expected stationary, but formal testing optional due to stability

- Differencing order: **d = 1** for TSLA and SPY.

---

## **6. Risk Metrics**

| Asset | Sharpe Ratio    | VaR (95%)        |
| ----- | --------------- | ---------------- |
| TSLA  | High (positive) | \~ -6% (daily)   |
| SPY   | Moderate        | \~ -2% (daily)   |
| BND   | \~0             | \~ -0.5% (daily) |

- **Sharpe Ratio**: TSLA has high potential risk-adjusted returns; BND offers stability but minimal reward.
- **Value at Risk (VaR)**: Reflects potential one-day loss at 95% confidence level.

---

## **7. Key Insights**

- TSLA offers the greatest potential reward but at the cost of high volatility and large drawdowns.
- BND provides portfolio stability with negligible daily losses.
- SPY delivers balanced market exposure with moderate risk and steady long-term growth.
- Both TSLA and SPY require differencing for ARIMA/SARIMA modeling.

---

## **8. Next Steps**

- Proceed to **Task 2**: Develop time series forecasting models for TSLA using both **ARIMA/SARIMA** and **LSTM**.
- Compare performance metrics (MAE, RMSE, MAPE) to select the best model for market trend forecasting.

---
