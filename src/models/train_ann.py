import joblib, os, yaml, json, datetime
import pandas as pd
import numpy as np

from typing import Tuple, Dict
import tensorflow as tf
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from dvclive import Live
from sklearn.model_selection import train_test_split


def main() -> None:

    # Load configuration
    with open("config/params.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Load data
    x_train, y_train, x_test, y_test = load_data(
        train_path=config["data"]["train_path"],
        test_path=config["data"]["test_path"],
        target_column=config["data"]["target_column"]
    )

    y_train_log = np.log1p(y_train)
    
    # Load dataset params
    dataset_params = config["models"]["ann"]["dataset_params"]

    train_ds, val_ds = create_tf_dataset(
        x_train=x_train.values,
        y_train=y_train_log,
        **dataset_params
        )

    # Load model params
    model_params = config["models"]["ann"]["model_params"]
    
    # Initialize model
    model: tf.keras.Model = create_ann_model(input_dim=x_train.shape[1], **model_params)

    os.makedirs("logs/fit/", exist_ok=True)
    log_dir: str = "logs/fit/" + datetime.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq="epoch",
        profile_batch=0
    )

    tf.summary.trace_on(graph=True, profiler=False)

    # Load train params
    train_params = config["models"]["ann"]["train_params"]

    with Live(dir="dvclive/ann", save_dvc_exp=True) as live:

        # Log parameters
        # live.log_param("model", "ann")
        # for param, value in dataset_params.items():
        #     live.log_param(param, value)
            
        # for param, value in model_params.items():
        #     live.log_param(param, value)
            
        # for param, value in train_params.items():
        #     live.log_param(param, value)
        

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=[tensorboard_callback],
            epochs=train_params["epochs"],
            verbose=train_params["verbose"]
        )

        y_pred: np.ndarray = np.expm1(model.predict(x_test).flatten())

        metrics: Dict[str, float] = compute_metrics(y_test.values, y_pred)

        for metric, value in metrics.items():
            live.log_metric(f"test/{metric}", value)

        # Learning Curves
        os.makedirs(config["reports"]["figures_path"], exist_ok=True)

        plt.figure(figsize=(12, 6))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend(["Train", "Validation"])
        plt.savefig("reports/figures/ann_loss_curve.png")

        # Weight Histograms
        for i, layer in enumerate(model.layers):
            if hasattr(layer, "kernel"):
                weights = layer.kernel.numpy().flatten()
                plt.figure(figsize=(12, 6))
                plt.hist(weights, bins=100)
                plt.title(f"Layer {i} Weight Distribution")
                plt.xlabel("Weight value")
                plt.ylabel("Frequency")
                plt.savefig(config["reports"]["figures_path"] + f"ann_weights_layer_{i}.png")
                plt.close()
        
        tf.keras.utils.plot_model(
            model,
            to_file=config["reports"]["figures_path"] + "ann_model_graph.png",
            show_shapes=True,
            show_layer_names=True,
            dpi=300
        )

        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.trace_export(
                name="ANN_graph_trace",
                step=0,
                profiler_outdir=log_dir
            )
        
        writer = tf.summary.create_file_writer(log_dir)

        with writer.as_default():

            # ---- 1. Скаляры теста (в исходной шкале) ----
            tf.summary.scalar("test/RMSE_original", metrics["rmse"], step=0)
            tf.summary.scalar("test/MAE_original", metrics["mae"], step=0)
            tf.summary.scalar("test/R2_original", metrics["r2"], step=0)

            # ---- 2. Распределение ошибок ----
            errors = y_test.values - y_pred
            tf.summary.histogram("test/error_distribution", errors, step=0)

            # ---- 3. Распределение предсказаний ----
            tf.summary.histogram("test/y_pred_distribution", y_pred, step=0)
            tf.summary.histogram("test/y_true_distribution", y_test.values, step=0)

            # ---- 4. Нормы весов по слоям ----
            for layer in model.layers:
                if hasattr(layer, "kernel"):
                    weights = layer.kernel
                    weight_norm = tf.norm(weights)
                    tf.summary.scalar(f"weights_norm/{layer.name}", weight_norm, step=0)
                    tf.summary.histogram(f"weights_hist/{layer.name}", weights, step=0)

            writer.flush()

    # Save Model
    os.makedirs(config["models"]["models_path"], exist_ok=True)
    model.save(config["models"]["models_path"] + "ann.keras")


# Model Creation
def create_ann_model(
    input_dim: int,
    n_hidden_layers: int,
    n_neurons: int,
    activation: str,
    learning_rate: float
) -> tf.keras.Model:

    # Input layer
    inputs: tf.keras.Input = tf.keras.Input(shape=(input_dim,), name="input_layer")

    # Hidden layers
    x = inputs

    for layer_idx in range(n_hidden_layers):
        x: tf.keras.layers.Dense = tf.keras.layers.Dense(units=n_neurons, activation=activation, kernel_initializer="he_normal", name=f"dense_hidden_{layer_idx+1}")(x)

    # Output layer (Regression)
    outputs: tf.Tensor = tf.keras.layers.Dense(units=1, activation="linear", name="output_layer")(x)

    # Model construction
    model: tf.keras.Model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ANN_regressor")

    # Compilation
    optimizer: tf.keras.optimizers.Adam = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.R2Score(name='r2_score'),
        ]
    )

    return model

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

    x_test= test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    return x_train, y_train, x_test, y_test

def create_tf_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float,
    batch_size: int,
    random_state: int = 42
) -> tf.data.Dataset:

    # Split to train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_split,
        random_state=random_state
        )

    # Make TF datasets
    train_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = (train_ds
                .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)       # Перемешивает выборку данных на каждой эпохе
                .batch(batch_size)                                                      # Разделяет на батчи
                .cache()                                                                # Устраняет повторное чтение данных при каждой эпохе
                .prefetch(tf.data.AUTOTUNE))                                            # готовит следующий batch параллельно обучению текущего
    
    val_ds: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = (val_ds
              .batch(batch_size)
              .cache()
              .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    """
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

if __name__ == "__main__":
    main()