"""Module providing network layer functionality."""

import tensorflow as tf


def dense(
    units, activation, regularizer="l2", regularizer_rate=0.001, max_norm=3
):
    """Helper function which creates a dense layer.

    :group: utils

    Parameters
    ----------

    units
        Number of nodes in the layer.
    activation
        Activation function to use in the layer.
    regularizer
        Enable regularization on the layer, using either "l1", "l2", or "l1l2".
    max_norm
        Enable the max norm constraint.

    """

    if regularizer == "l1":
        regularizer = tf.keras.regularizers.l1(regularizer_rate)
    elif regularizer == "l2":
        regularizer = tf.keras.regularizers.l2(regularizer_rate)
    elif regularizer == "l1l2":
        regularizer = tf.keras.regularizers.L1L2(regularizer_rate)

    return tf.keras.layers.Dense(
        units,
        activation=activation,
        # activity_regularizer=regularizer,
        kernel_regularizer=regularizer,
        # bias_regularizer=regularizer,
        kernel_constraint=tf.keras.constraints.max_norm(max_norm),
        bias_constraint=tf.keras.constraints.max_norm(max_norm),
    )
