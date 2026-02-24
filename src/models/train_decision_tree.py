import joblib, os, yaml, json
import pandas as pd
import numpy as np

from typing import Tuple
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dvclive import Live


def main() -> None:

    # Load configuration
    with open('config/params.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_train, y_train, x_test, y_test = load_data(
        train_path=config["data"]["train_path"],
        test_path=config["data"]["test_path"],
        target_column=config["data"]["target_column"]
    )

    y_train_log = np.log1p(y_train)

    # Load model params
    params = config['models']['decision_tree']
    
    # Initialize model
    model = DecisionTreeRegressor(**params)

    with Live(dir="dvclive/decision_tree", save_dvc_exp=True) as live:

        # Log parameters
        # live.log_param("model", "DecisionTreeRegressor")
        # for param, value in params.items():
        #     live.log_param(param, value)

        # Train
        model.fit(x_train, y_train_log)

        # Predict
        y_pred = np.expm1(model.predict(x_test))   # inverse log transform

        # Metrics
        metrics = compute_metrics(y_test.values, y_pred)

        for metric, value in metrics.items():
            live.log_metric(f"test/{metric}", value)

        # Save tree plot
        os.makedirs(config["reports"]["figures_path"], exist_ok=True)
        plt.figure(figsize=(12, 6))
        
        plot_tree(model, max_depth=2, filled=True)
        plt.savefig(config["reports"]["figures_path"] + "decision_tree_structure.png")
        plt.close()

    # Save model
    os.makedirs(config['models']['models_path'], exist_ok=True)
    joblib.dump(model, config['models']['models_path'] + "decision_tree.pkl")


def load_data(
    train_path: str,
    test_path: str,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load train and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    x_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return x_train, y_train, x_test, y_test

def compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Compute regression metrics.
    """
    return {
        "rmse": float(root_mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred))
    }

if __name__ == "__main__":
    main()