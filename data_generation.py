import numpy as np
import pandas as pd


def generate_data(n_steps=5000):
    t = np.arange(n_steps)
    seasonal1 = np.sin(2 * np.pi * t / 24) # daily seasonality
    seasonal2 = np.sin(2 * np.pi * t / (24*7)) # weekly seasonality
    trend = t * 0.0005
    noise = np.random.normal(0, 0.1, n_steps)


    feature1 = seasonal1 + trend + noise
    feature2 = seasonal2 + noise
    feature3 = seasonal1 * 0.5 + seasonal2 * 0.2 + noise


    df = pd.DataFrame({"f1": feature1, "f2": feature2, "f3": feature3})
    df.to_csv("timeseries.csv", index=False)


if __name__ == "__main__":
    generate_data()