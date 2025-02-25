import lightgbm as lgb
import optuna
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import argparse
import src.utils.misc as misc
from src.dataset.synthetic_data import create_synthetic_data_sklearn

def main(args):
    misc.set_random_seeds(args.random_seed)

    if args.data_path is None:
        X_train, X_val, X_test, y_train, y_val, y_test = create_synthetic_data_sklearn(args.n_features,
                                                                                    args.n_informative, 
                                                                                    args.n_redundant, 
                                                                                    args.n_repeated,
                                                                                    args.n_classes, 
                                                                                    args.n_samples, 
                                                                                    args.n_clusters_per_class, 
                                                                                    args.class_sep, 
                                                                                    args.flip_y, 
                                                                                    args.random_seed, 
                                                                                    args.data_folder,
                                                                                    test_size=args.test_size,
                                                                                    val_size=args.val_size)
    else:
        data_folder = args.data_folder
        data = np.load(os.path.join(data_folder, args.data_path))
        X_train, X_val, X_test, y_train, y_val, y_test = data['X_train'], data['X_val'], data['X_test'], data['y_train'], data['y_val'], data['y_test']

    def objective(trial):
        param_grid = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'binary_error'],
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'max_bin': trial.suggest_int('max_bin', 200, 300)
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=0),
            optuna.integration.LightGBMPruningCallback(trial, 'binary_logloss')
        ]
        
        model = lgb.train(
            param_grid, 
            train_data,
            valid_sets=[valid_data],
            callbacks=callbacks,
            num_boost_round=1000
        )
        
        if hasattr(model, 'best_score'):
            return -model.best_score['valid_0']['binary_logloss']
        
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred_binary)
        return -accuracy

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    study.optimize(objective, n_trials=100, timeout=3600)  # 1 hour timeout

    print("Best Hyperparameters:")
    print(study.best_params)

    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['metric'] = ['binary_logloss', 'binary_error']

    final_model = lgb.train(
        best_params, 
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=1)],
        num_boost_round=1000
    )

    model_setting = f'final_model_n_feat{args.n_features}_n_informative{args.n_informative}_n_redundant{args.n_redundant}_n_repeated{args.n_repeated}_n_classes{args.n_classes}_n_samples{args.n_samples}_n_clusters_per_class{args.n_clusters_per_class}_class_sep{args.class_sep}_flip_y{args.flip_y}_random_state{args.random_seed}'
    final_model.save_model(os.path.join(args.model_folder, model_setting))

    final_preds = final_model.predict(X_test)
    final_preds_binary = (final_preds > 0.5).astype(int)
    final_accuracy = accuracy_score(y_test, final_preds_binary)
    print(f"Final Model Accuracy on test set: {final_accuracy:.4f}")

    importance = final_model.feature_importance(importance_type='gain')
    feature_names = [f'feature_{i}' for i in range(args.n_features)]
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f'{name}: {imp}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_informative", type=int, default=15)
    parser.add_argument("--n_clusters_per_class", type=int, default=2)
    parser.add_argument("--class_sep", type=float, default=1.0)
    parser.add_argument("--flip_y", type=float, default=0.01)
    parser.add_argument("--n_redundant", type=int, default=5)
    parser.add_argument("--n_repeated", type=int, default=0)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=1000000)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--data_folder", type=str, default="/home/grotehans/xai_locality/data/synthetic_data")
    parser.add_argument("--model_folder", type=str, default="/home/grotehans/xai_locality/models/synthetic_data")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.4)
    parser.add_argument("--val_size", type=float, default=0.1)
    args = parser.parse_args()
    main(args)