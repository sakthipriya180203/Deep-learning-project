import optuna
from dataset import load_data
import tensorflow as tf


X_train, X_test, y_train, y_test, _ = load_data()


def objective(trial):
    units = trial.suggest_int("units", 32, 128)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)


    model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units, input_shape=X_train.shape[1:]),
    tf.keras.layers.Dense(1)
    ])


    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model.evaluate(X_test, y_test, verbose=0)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print(study.best_params)