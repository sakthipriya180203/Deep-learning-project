import numpy as np

def generate_data(samples=1000, timesteps=20, features=8):
    X = np.random.rand(samples, timesteps, features)
    y = (np.mean(X, axis=(1, 2)) > 0.5).astype(int)
    return X, y

if __name__ == "__main__":
    X, y = generate_data()

    np.save("X.npy", X)
    np.save("y.npy", y)

    print("Dataset saved: X.npy, y.npy")
