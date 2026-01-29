import matplotlib.pyplot as plt
import tensorflow as tf
from dataset import load_data
from model_attention import build_attention_model


X_train, X_test, y_train, y_test, _ = load_data()
model = build_attention_model(X_train.shape[1:])
model.load_weights("attention_model.h5")


# Get attention weights manually (advanced step for report)
print("Attention visualization requires custom extraction - mention conceptually in report")