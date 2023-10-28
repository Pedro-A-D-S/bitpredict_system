import tensorflow as tf


class NBeatsBlock(tf.keras.layers.Layer):
    """
    A NBeats block.

    Args:
        input_size: The size of the input to the block.
        theta_size: The size of the output theta layer.
        horizon: The prediction horizon.
        n_neurons: The number of neurons in each hidden layer.
        n_layers: The number of hidden layers.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        n_neurons: int,
        n_layers: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [
            tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)
        ]

        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(
            theta_size, activation="linear", name="theta"
        )

    def call(self, inputs):
        """
        Forward pass of the NBeats block.

        Args:
            inputs: The input tensor to the block.

        Returns:
            The backcast and forecast tensors.
        """

        x = inputs

        for layer in self.hidden:
            # pass inputs through each hidden layer
            x = layer(x)

        theta = self.theta_layer(x)

        # Output the backcast and forecast from theta
        backcast, forecast = theta[:, : self.input_size], theta[:, -self.horizon :]

        return backcast, forecast
