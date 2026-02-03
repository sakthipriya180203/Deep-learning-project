import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline(window):
    model = models.Sequential([
        layers.Input(shape=(window,1)),
        layers.LSTM(64),
        layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model
