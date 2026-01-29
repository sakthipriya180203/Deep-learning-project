import tensorflow as tf


def build_baseline(input_shape):
    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=input_shape),
    tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model