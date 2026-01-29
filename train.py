from dataset import load_data
from model_baseline import build_baseline
from model_attention import build_attention_model


X_train, X_test, y_train, y_test, _ = load_data()


# Baseline
baseline = build_baseline(X_train.shape[1:])
baseline.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
baseline.save("baseline_model.h5")


# Attention Model
attention = build_attention_model(X_train.shape[1:])
attention.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
attention.save("attention_model.h5")