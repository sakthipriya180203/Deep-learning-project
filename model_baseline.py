import tensorflow as tf
from tensorflow.keras import layers

def build_baseline(window=20, features=3, units=64):
    model = tf.keras.Sequential([layers.LSTM(units, input_shape=(window, features)),layers.Dropout(0.2),layers.Dense(1)]) 
    model.compile(optimizer="adam", loss="mse")
    return model

