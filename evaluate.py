import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
from dataset import load_data
from model_attention import AttentionLayer   # IMPORTANT LINE

# Load data
X_train, X_test, y_train, y_test, _ = load_data()

# Load models with custom layer
baseline = load_model("baseline_model.h5", compile=False)

attention = load_model(
    "attention_model.h5",
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

# Predictions
pred_base = baseline.predict(X_test)
pred_attn = attention.predict(X_test)

# Metrics
def metrics(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    mae = mean_absolute_error(y, p)
    mape = np.mean(np.abs((y - p) / y)) * 100
    return rmse, mae, mape

print("Baseline Model Metrics:")
print("RMSE, MAE, MAPE =", metrics(y_test, pred_base))

print("\nAttention Model Metrics:")
print("RMSE, MAE, MAPE =", metrics(y_test, pred_attn))
