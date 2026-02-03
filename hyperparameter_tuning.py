import optuna
from dataset import load_data
from model_attention import build_attention_model


X_train, y_train, _, _, _ = load_data()


def objective(trial):
    batch = trial.suggest_categorical("batch", [16,32,64])

    model = build_attention_model(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch,
        validation_split=0.2,
        verbose=0
    )

    return min(history.history["val_loss"])


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

print(study.best_params)
