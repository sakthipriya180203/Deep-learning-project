import numpy as np
import pandas as pd

def generate_data(N=5000):
    np.random.seed(42)

    
    t = np.arange(N)
    trend = 0.002 * t
    season1 = 5 * np.sin(2*np.pi*t/24)
    season2 = 10 * np.sin(2*np.pi*t/365)
    noise = np.random.normal(0, 2, N)

    f1 = trend + season1 + season2 + noise
    f2 = 0.5*f1 + np.random.normal(0,1,N)
    f3 = np.random.normal(10,3,N)

    df = pd.DataFrame({"f1":f1,"f2":f2,"f3":f3})
    df.to_csv("timeseries.csv", index=False)
    print("Dataset saved as timeseries.csv")


if __name__ == "__main__":
    generate_data()
