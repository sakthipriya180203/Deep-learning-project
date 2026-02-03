from dataset import load_data
from model_baseline import build_baseline
from model_attention import build_attention

X_train, X_test, y_train, y_test = load_data()

baseline = build_baseline()
baseline.fit(X_train, y_train, epochs=10, batch_size=64)

baseline.save("baseline_model.h5")

att_model, _ = build_attention()
att_model.fit(X_train, y_train, epochs=10, batch_size=64)

att_model.save("attention_model.h5")

print("Models saved!")
