import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataset import load_data
import tensorflow as tf

X_train, X_test, y_train, y_test = load_data()

baseline = tf.keras.models.load_model("baseline_model.h5")
attention = tf.keras.models.load_model("attention_model.h5")

pred_base = baseline.predict(X_test)
pred_att = attention.predict(X_test)

print("Baseline RMSE:", np.sqrt(mean_squared_error(y_test, pred_base)))
print("Attention RMSE:", np.sqrt(mean_squared_error(y_test, pred_att)))
print("Baseline MAE:", mean_absolute_error(y_test, pred_base))
print("Attention MAE:", mean_absolute_error(y_test, pred_att))
