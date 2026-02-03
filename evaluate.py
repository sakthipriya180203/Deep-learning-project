import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from dataset import load_data


def mape(y, yhat):
    return np.mean(np.abs((y-yhat)/y))*100


X_train, y_train, X_test, y_test, scaler = load_data()

model = load_model("attention_model.h5", compile=False)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
m = mape(y_test, pred)

print("RMSE:", rmse)
print("MAE :", mae)
print("MAPE:", m)
