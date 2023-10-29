"""Utils functions for notebooks"""
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from typing import List, Tuple, Dict
from tensorflow.keras import layers


def split_data(
    timesteps: List[float], prices: List[float], split_ratio: float = 0.8
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Split time series data into training and testing sets.

    Args:
        timesteps (List[float]): List of timestamps or time-related data.
        prices (List[float]): List of corresponding prices or target values.
        split_ratio (float, optional): The ratio of data to be used for training. Default is 0.8 (80%).

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: A tuple containing:
            - X_train (List[float]): Training data for timesteps.
            - y_train (List[float]): Training data for prices.
            - X_test (List[float]): Testing data for timesteps.
            - y_test (List[float]): Testing data for prices.
    """
    split_size = int(split_ratio * len(prices))

    X_train, y_train = timesteps[:split_size], prices[:split_size]
    X_test, y_test = timesteps[split_size:], prices[split_size:]

    return X_train, X_test, y_train, y_test


def plot_time_series(
    timesteps: List[float],
    values: List[float],
    format: str = ".",
    start: int = 0,
    end: Optional[int] = None,
    label: Optional[str] = None,
) -> None:
    """
    Plots a series of timesteps against corresponding values.

    Parameters:
        timesteps (List[float]): A list of timestamps or time-related data.
        values (List[float]): A list of corresponding values across timesteps.
        format (str, optional): The style of the plot. Default is ".".
        start (int, optional): The index to start the plot. Defaults to 0.
        end (int, optional): The index to end the plot. Defaults to None (end of the data).
        label (str, optional): A label to show on the plot of values.
    """
    # Plot the series
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("BTC Price")
    if label:
        plt.legend(fontsize=14)  # Make the label bigger
    plt.grid(True)


def save_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: str,
) -> None:
    """
    Save NumPy arrays X_train, X_test, y_train, and y_test to CSV files in the specified directory.

    Parameters:
        X_train (np.ndarray): Training data for input features.
        X_test (np.ndarray): Testing data for input features.
        y_train (np.ndarray): Training data for target values.
        y_test (np.ndarray): Testing data for target values.
        save_dir (str): The directory where CSV files will be saved.

    Returns:
        None
    """
    np.save(f"{save_dir}/X_train", X_train, allow_pickle=False)
    np.save(f"{save_dir}/X_test", X_test, allow_pickle=False)
    np.save(f"{save_dir}/y_train", y_train, allow_pickle=False)
    np.save(f"{save_dir}/y_test", y_test, allow_pickle=False)


def load_data(save_dir: str) -> tuple:
    """
    Load NumPy arrays from CSV files in the specified directory.

    Parameters:
        save_dir (str): The directory where CSV files are saved.

    Returns:
        X_train, X_test, y_train, y_test (np.ndarray): Training and testing data for input features and target values.
    """

    X_train = np.load(f"{save_dir}/X_train.npy")
    X_test = np.load(f"{save_dir}/X_test.npy")
    y_train = np.load(f"{save_dir}/y_train.npy")
    y_test = np.load(f"{save_dir}/y_test.npy")

    return X_train, X_test, y_train, y_test


def load_btc_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load Bitcoin price data from a CSV file.

    Returns:
        bitcoin_prices (pd.DataFrame): A DataFrame containing the Bitcoin price data.
        timesteps (np.ndarray): A NumPy array containing the timesteps.
        prices (np.ndarray): A NumPy array containing the prices.
    """

    df = pd.read_csv(
        "../data/raw/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv",
        parse_dates=["Date"],
        index_col=["Date"],
    )

    bitcoin_prices = pd.DataFrame(df["Closing Price (USD)"]).rename(
        columns={"Closing Price (USD)": "Price"}
    )

    timesteps = bitcoin_prices.index.to_numpy()
    prices = bitcoin_prices["Price"].to_numpy()

    return bitcoin_prices, timesteps, prices


def mean_absolute_scaled_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    """
    Calculates the mean absolute scaled error (MASE) between two tensors.

    Args:
        y_true: The ground truth tensor.
        y_pred: The predicted tensor.

    Returns:
        The MASE score, a float value.

    Notes:
        MASE is defined as the mean absolute error (MAE) of the forecast divided by the MAE of a naive forecast. The naive forecast is simply the previous value of the time series. A lower MASE score indicates a better forecast.

        This function assumes that the time series has no seasonality.
    """

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))

    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))

    return mae / mae_naive_no_season


def evaluate_preds(y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, float]:
    """
    Evaluates the performance of a time series forecast.

    Args:
        y_true: The ground truth tensor.
        y_pred: The predicted tensor.

    Returns:
        A dictionary of evaluation metrics, including MAE, MSE, RMSE, MAPE, and MASE.
    """
    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    # puts and emphasis on outliers (all errors get squared)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy(),
    }


def get_labelled_windows(
    x: np.ndarray, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates labels for windowed dataset.

    Args:
        x: A 1D NumPy array.
        horizon: The number of steps to predict into the future. Defaults to 1.

    Returns:
        A tuple of two NumPy arrays, where the first array contains the windows and the second array contains the labels.

    Examples:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> windows, labels = get_labelled_windows(x)
        >>> windows
        array([[1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6]])
        >>> labels
        array([6])
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(
    x: np.ndarray, window_size: int = 7, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.

    Args:
        x: A 1D NumPy array.
        window_size: The size of the windows to create. Defaults to 7.
        horizon: The number of steps to predict into the future. Defaults to 1.

    Returns:
        A tuple of two NumPy arrays, where the first array contains the windows and the second array contains the labels.

    Examples:
        >>> x = np.array([1, 2, 3, 4, 5, 6])
        >>> windows, labels = make_windows(x)
        >>> windows
        array([[1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6]])
        >>> labels
        array([6])
    """

    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = (
        window_step
        + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    )

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels


def make_train_test_splits(
    windows: np.ndarray, labels: np.ndarray, test_split: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits matching pairs of windows and labels into train and test splits.

    Args:
        windows: A 2D NumPy array of windows.
        labels: A 1D NumPy array of labels.
        test_split: The proportion of the data to use for the test set. Defaults to 0.2.

    Returns:
        A tuple of four NumPy arrays, where the first array contains the train windows, the second array contains the train labels, the third array contains the test windows, and the fourth array contains the test labels.

    Examples:
        >>> windows = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
        >>> labels = np.array([6])
        >>> train_windows, test_windows, train_labels, test_labels = make_train_test_splits(windows, labels)
        >>> train_windows
        array([[1, 2, 3, 4, 5]])
        >>> train_labels
        array([6])
        >>> test_windows
        array([[2, 3, 4, 5, 6]])
        >>> test_labels
        array([6])
    """

    split_size = int(len(windows) * (1 - test_split))

    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]

    return train_windows, test_windows, train_labels, test_labels


def create_model_checkpoint(
    model_name: str, save_path: str = "../data/model_experiments"
) -> tf.keras.callbacks.ModelCheckpoint:
    """
    Creates a ModelCheckpoint callback to save the best model during training.

    Args:
        model_name: The name of the model to save.
        save_path: The path to save the model to. Defaults to "model_experiments".

    Returns:
        A ModelCheckpoint callback.
    """

    filepath = os.path.join(save_path, model_name)
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        verbose=0,
        save_best_only=True,
    )


def make_preds(model: tf.keras.Model, input_data: np.ndarray) -> np.ndarray:
    """
    Uses a model to make predictions on input_data.

    Args:
        model: A trained model.
        input_data: Windowed input data (same kind of data model was trained on).

    Returns:
        Model predictions on input_data.
    """

    forecast = model.predict(input_data)
    return tf.squeeze(forecast)


def evaluate_tf_preds(y_true: tf.Tensor, y_pred: tf.Tensor) -> dict[str, float]:
    """
    Evaluates TensorFlow predictions.

    Args:
        y_true: A NumPy array or TensorFlow tensor of ground truth values.
        y_pred: A NumPy array or TensorFlow tensor of predicted values.

    Returns:
        A dictionary of evaluation metrics, including MAE, MSE, RMSE, MAPE, and MASE.
    """

    # Make sure float32 (for metric calculations)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    mase = mean_absolute_scaled_error(y_true, y_pred)

    # Account for different sized metrics (for longer horizons, reduce to single number)
    if (
        mae.ndim > 0
    ):  # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        mase = tf.reduce_mean(mase)

    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy(),
    }


def get_ensemble_models(
    horizon: int,
    train_data: tf.keras.utils.Sequence,
    test_data: tf.keras.utils.Sequence,
    num_iter: int = 10,
    num_epochs: int = 100,
    loss_fns: List[str] = ["mae", "mse", "mape"],
) -> List[tf.keras.Model]:
    """
    Returns a list of num_iter models each trained on MAE, MSE and MAPE loss.

    Args:
        horizon: The prediction horizon.
        train_data: A TensorFlow sequence of training data.
        test_data: A TensorFlow sequence of test data.
        num_iter: The number of models to train.
        num_epochs: The number of epochs to train each model for.
        loss_fns: A list of loss functions to train the models on.

    Returns:
        A list of trained models.
    """

    # Make empty list for trained ensemble models
    ensemble_models = []

    # Create num_iter number of models per loss function
    for i in range(num_iter):
        # Build and fit a new model with a different loss function
        for loss_function in loss_fns:
            print(
                f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}"
            )

            # Construct a simple model (similar to model_1)
            model = tf.keras.Sequential(
                [
                    # Initialize layers with normal (Gaussian) distribution so we can use the models for prediction
                    # interval estimation later: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
                    layers.Dense(
                        128, kernel_initializer="he_normal", activation="relu"
                    ),
                    layers.Dense(
                        128, kernel_initializer="he_normal", activation="relu"
                    ),
                    layers.Dense(horizon),
                ]
            )

            # Compile simple model with current loss function
            model.compile(
                loss=loss_function,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae", "mse"],
            )

            # Fit model
            model.fit(
                train_data,
                epochs=num_epochs,
                verbose=0,
                validation_data=test_data,
                # Add callbacks to prevent training from going/stalling for too long
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=200, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", patience=100, verbose=1
                    ),
                ],
            )

            # Append fitted model to list of ensemble models
            ensemble_models.append(model)

    return ensemble_models  # return list of trained models


def make_ensemble_preds(
    ensemble_models: List[tf.keras.Model], data: tf.Tensor
) -> tf.Tensor:
    """
    Makes predictions on a given dataset using an ensemble of models.

    Args:
        ensemble_models: A list of trained ensemble models.
        data: The dataset to make predictions on.

    Returns:
        A Tensor of ensemble predictions.
    """

    ensemble_preds = []
    for model in ensemble_models:
        # make predictions with current ensemble model
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return tf.constant(tf.squeeze(ensemble_preds))


def get_upper_lower(preds: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Calculates the upper and lower bounds of a prediction interval.

    Args:
        preds: A Tensor of predictions from multiple ensemble models.

    Returns:
        A tuple of two Tensors, where the first Tensor contains the lower bounds and the second Tensor contains the upper bounds.
    """

    # 1. Take the predictions of multiple randomly initialized deep learning neural networks

    # 2. Measure the standard deviation of the predictions
    std = tf.math.reduce_std(preds, axis=0)

    # 3. Multiply the standard deviation by 1.96
    interval = 1.96 * std  # https://en.wikipedia.org/wiki/1.96

    # 4. Get the prediction interval upper and lower bounds
    preds_mean = tf.reduce_mean(preds, axis=0)
    lower, upper = preds_mean - interval, preds_mean + interval
    return lower, upper


def make_future_forecast(
    values: tf.Tensor, model: tf.keras.Model, into_future: int, window_size: int
) -> list[float]:
    """
    Makes future forecasts into_future steps after values ends.

    Args:
        values: A Tensor of historical values.
        model: A trained TensorFlow model.
        into_future: The number of future steps to forecast.
        window_size: The size of the window used to train the model.

    Returns:
        A list of future forecasts, as floats.
    """

    # Make an empty list for future forecasts/prepare data to forecast on
    future_forecast = []
    # only want preds from the last window (this will get updated)
    last_window = values[-window_size:]

    # Make INTO_FUTURE number of predictions, altering the data which gets predicted on each time
    for _ in range(into_future):
        # Predict on last window then append it again, again, again (model starts to make forecasts on its own forecasts)
        future_pred = model.predict(tf.expand_dims(last_window, axis=0))
        print(
            f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(future_pred).numpy()}\n"
        )

        # Append predictions to future_forecast
        future_forecast.append(tf.squeeze(future_pred).numpy())

        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, future_pred)[-window_size:]

    return future_forecast


def get_future_dates(
    start_date: np.datetime64,
    into_future: int,
    offset: int = 1,
) -> np.ndarray:
    """
    Returns an array of datetime values ranging from start_date to start_date+horizon.

    Args:
        start_date: The date to start the range from (np.datetime64).
        into_future: The number of days to add onto start date for the range (int).
        offset: The number of days to offset start_date by (default 1).

    Returns:
        A NumPy array of datetime values.
    """

    # specify start date, "D" stands for day
    start_date = start_date + np.timedelta64(offset, "D")
    end_date = start_date + np.timedelta64(into_future, "D")  # specify end date
    # return a date range between start date and end date
    return np.arange(start_date, end_date, dtype="datetime64[D]")
