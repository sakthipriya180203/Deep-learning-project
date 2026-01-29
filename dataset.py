import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def create_sequences(data, seq_len=50):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 0]) # predict f1
    return np.array(X), np.array(y)


def load_data(seq_len=50):
    df = pd.read_csv("timeseries.csv")
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)


    X, y = create_sequences(data, seq_len)
    split = int(len(X) * 0.8)


    return X[:split], X[split:], y[:split], y[split:], scaler