import numpy as np
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2):
    X = np.load("X.npy")
    y = np.load("y.npy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, y_train, X_test, y_test
