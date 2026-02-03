import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from dataset import load_data
from model_attention import AttentionLayer

model = load_model(
    "attention_model.keras",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

X_train, y_train, X_test, y_test = load_data()

sample = X_test[0:1]

prediction = model.predict(sample)

print("Prediction:", prediction)

plt.plot(sample.flatten())
plt.title("Input sequence")
plt.show()
