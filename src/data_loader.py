import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def download_data(ticker="AAPL", start="2013-01-01", end="2023-01-01"):
    df = yf.download(ticker, start=start, end=end)[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

def preprocess_data(df, window_size=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    windows = []
    for i in range(len(scaled_data) - window_size):
        windows.append(scaled_data[i:i+window_size])

    return np.array(windows), scaler