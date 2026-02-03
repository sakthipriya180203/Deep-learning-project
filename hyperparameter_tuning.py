import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from dataset import load_data
from model_attention import build_attention

X_train, X_test, y_train, y_test = load_data()

def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)


    model, _ = build_attention(units=units, lr=lr)
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)

    pred = model.predict(X_test, verbose=0)
    return np.sqrt(mean_squared_error(y_test, pred))


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print("Best params:", study.best_params)
