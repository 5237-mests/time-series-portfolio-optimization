# Week 11 KAIM

## 📌 Project Overview

This project is part of the **10 Academy KAIM Week 11 Challenge**:  
**Time Series Forecasting for Portfolio Management Optimization**.  
The goal is to apply time series forecasting techniques to financial data (TSLA, BND, SPY) to inform portfolio allocation strategies based on Modern Portfolio Theory (MPT).

## 📅 Task 1 Objective

Task 1 focuses on:

- Data extraction using Yahoo Finance API (`yfinance`)
- Data cleaning and preparation
- Exploratory Data Analysis (EDA)
- Stationarity testing
- Volatility and risk metrics calculation
- Outlier detection

## 📊 Assets Analyzed

| Asset | Description                              |
| ----- | ---------------------------------------- |
| TSLA  | High-growth, high-volatility stock       |
| BND   | Bond ETF, stability & low risk           |
| SPY   | S&P 500 ETF, diversified market exposure |

## ✅ Completed Steps

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

## 📈 Key Insights

- TSLA: High potential reward, high volatility
- BND: Stability with minimal daily loss
- SPY: Balanced exposure, moderate risk
- Differencing needed for TSLA & SPY before ARIMA/SARIMA modeling
