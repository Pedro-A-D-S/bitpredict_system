import bentoml
import yaml
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from utils import load_data, load_btc_data, make_train_test_splits, make_windows


def read_config_file(filename: str) -> dict:
    """
    Read and parse a YAML configuration file.

    Parameters:
        filename (str): The name of the configuration file (without the extension).

    Returns:
        dict: The configuration dictionary.
    """
    with open(f'../conf/{filename}.yaml') as f:
        config = yaml.safe_load(f)
    return config


def load_split_data(config: dict) -> tuple:
    """
    Load and split data using the specified configuration.

    Parameters:
        config (dict): Configuration dictionary containing 'save_dir' for data location.

    Returns:
        tuple: A tuple of X_train, X_test, y_train, and y_test arrays.
    """
    X_train, X_test, y_train, y_test = load_data(save_dir=config['save_dir'])
    print("Loaded X_train, X_test, y_train, y_test data successfully!")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


def load_bitcoin_data() -> tuple:
    """
    Load Bitcoin-related data.

    Returns:
        tuple: A tuple of bitcoin_prices, timesteps, and prices.
    """
    bitcoin_prices, timesteps, prices = load_btc_data()
    print("Loaded bitcoin_prices, timesteps, prices data successfully!")
    return bitcoin_prices, timesteps, prices


def making_windows(config: dict, prices: list) -> tuple:
    """
    Create data windows for training.

    Parameters:
        config (dict): Configuration dictionary containing 'WINDOW_SIZE' and 'HORIZON'.
        prices (list): List of prices for creating windows.

    Returns:
        tuple: A tuple of full_windows and full_labels arrays.
    """
    full_windows, full_labels = make_windows(
        prices,
        window_size=config['WINDOW_SIZE'],
        horizon=config['HORIZON']
    )
    print("Made windows successfully!")
    print(len(full_windows), len(full_labels))
    return full_windows, full_labels


def train_test_split(full_windows: list, full_labels: list) -> tuple:
    """
    Split data into training and testing sets.

    Parameters:
        full_windows (list): List of data windows.
        full_labels (list): List of labels.

    Returns:
        tuple: A tuple of train_windows, test_windows, train_labels, and test_labels arrays.
    """
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        full_windows,
        full_labels)
    print("Made train/test splits successfully!")
    return train_windows, test_windows, train_labels, test_labels


def build_model(config: dict) -> Sequential:
    """
    Build a Keras Sequential model.

    Parameters:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        Sequential: The Keras Sequential model.
    """
    tf.random.set_seed(42)
    
    model = Sequential(
        [
            Dense(config['n_units'], activation=config['hidden_layer']),
            Dense(config['HORIZON'], activation=config['output_layer']),
        ]
    )
    print("Model created successfully!")
    
    model.compile(
        loss=config['loss'],
        optimizer=Adam(),
        metrics=config['metrics']
    )
    print("Model compiled successfully!")
    
    return model
    


def fit_model(
    model,
    train_windows: list,
    train_labels: list,
    test_windows: list,
    test_labels: list,
    config: dict
    ) -> Sequential:
    """
    Train a Keras model with the given data.

    Parameters:
        model (Sequential): The Keras Sequential model to train.
        train_windows (list): List of training data windows.
        train_labels (list): List of training labels.
        test_windows (list): List of testing data windows.
        test_labels (list): List of testing labels.
        config (dict): Configuration dictionary containing training parameters.

    Returns:
        None
    """
    model.fit(
        x=train_windows,
        y=train_labels,
        epochs=config['epochs'],
        verbose=1,
        validation_data=(test_windows, test_labels),
    )
    print("Model fitted!")
    print(model.summary())
    print(model)
    return model


def save_model(model: Sequential, config: dict) -> None:
    """
    Save a Keras model using BentoML.

    Parameters:
        model (Sequential): The Keras Sequential model to save.
        config (dict): Configuration dictionary containing the model name.

    Returns:
        None
    """
    bentoml.tensorflow.save_model(
        model,
        config['model_name'],
        signatures={"__call__": {"batchable": True, "batch_dim": 0}}
    )
    print("Saved Bento model!")


def train():
    config = read_config_file("config")
    X_train, X_test, y_train, y_test = load_split_data(config)
    bitcoin_prices, timesteps, prices = load_bitcoin_data()
    full_windows, full_labels = making_windows(config, prices)
    train_windows, test_windows, train_labels, test_labels = train_test_split(
        full_windows,
        full_labels)
    model = build_model(config)
    model = fit_model(
        model,
        train_windows,
        train_labels,
        test_windows,
        test_labels,
        config)
    save_model(model, config)


if __name__ == "__main__":
    train()
