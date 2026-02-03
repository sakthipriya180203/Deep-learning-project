import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

WINDOW = 20

def load_data():
    df = pd.read_csv("timeseries.csv")

    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled)-WINDOW):
        X.append(scaled[i:i+WINDOW])
        y.append(scaled[i+WINDOW,0])

    X = np.array(X)
    y = np.array(y)

    split = int(0.8*len(X))
    return X[:split], X[split:], y[:split], y[split:]