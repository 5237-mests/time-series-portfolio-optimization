import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker: str, start: str, end: str, save_path: str = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    df.reset_index(inplace=True)
    df['Ticker'] = ticker
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df

def load_assets():
    tickers = {
        'TSLA': '../data/raw/tsla.csv',
        'BND': '../data/raw/bnd.csv',
        'SPY': '../data/raw/spy.csv'
    }
    start = "2015-07-01"
    end = "2025-07-31"

    all_data = {}
    for ticker, path in tickers.items():
        all_data[ticker] = fetch_data(ticker, start, end, path)
    return all_data
