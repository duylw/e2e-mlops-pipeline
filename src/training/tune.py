import optuna
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error


def finetune_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """
    Finds optimal hyperparameters for XGBoost using Optuna early stopping trials.
    """
    print(f"Starting Optuna tuning with {n_trials} trials...")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'n_jobs': -1,
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0)
        }

        model = xgb.XGBRegressor(**params, n_estimators=1000, early_stopping_rounds=100)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        trial.set_user_attr('best_iteration', model.best_iteration)
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        return rmse

    study = optuna.create_study(
        direction='minimize',
        study_name="XGBoost_Taxi_Duration",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']
    
    # Add a small buffer since we train on more data (train+val) for the final model
    best_params['n_estimators'] = int(best_iteration * 1.1) + 1

    print("\nOPTIMAL TUNING RESULTS:")
    print(f"Best Val RMSE: {study.best_value:.4f}")
    print(f"Best n_estimators (early stopping): {best_iteration}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return best_params, study
