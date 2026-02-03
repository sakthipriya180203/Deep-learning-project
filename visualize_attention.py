import matplotlib.pyplot as plt
from dataset import load_data
from model_attention import build_attention

X_train, X_test, y_train, y_test = load_data()

model, weight_model = build_attention()
model.load_weights("attention_model.h5")

weights = weight_model.predict(X_test[:1])[0]

plt.imshow(weights.reshape(1,-1), aspect='auto')
plt.colorbar()
plt.title("Attention Weights")
plt.show()
