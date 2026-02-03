import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from dataset import load_data
from model_attention import AttentionLayer

X_train, y_train, X_test, y_test = load_data()

inputs = Input(shape=(20, 8))
x = LSTM(64, return_sequences=True)(inputs)
x = AttentionLayer()(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("attention_model.keras")

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
