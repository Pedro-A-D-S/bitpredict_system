'''Utils functions for notebooks'''
from typing import List, Optional
import matplotlib as plt
import numpy as np
from typing import List, Tuple


def split_data(
    timesteps: List[float],
    prices: List[float],
    split_ratio: float = 0.8
) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
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
    '''
    split_size = int(split_ratio * len(prices))

    X_train, y_train = timesteps[:split_size], prices[:split_size]
    X_test, y_test = timesteps[split_size:], prices[split_size:]

    return X_train, X_test, y_train, y_test


def plot_time_series(
    timesteps: List[float],
    values: List[float],
    format: str = '.',
    start: int = 0,
    end: Optional[int] = None,
    label: Optional[str] = None
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
    save_dir: str
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
