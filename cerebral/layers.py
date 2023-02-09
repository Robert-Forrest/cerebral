"""Module providing network layer functionality."""

import tensorflow as tf


def dense(
    units, activation, regularizer=None, regularizer_rate=None, max_norm=None
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
    else:
        regularizer = None

    return tf.keras.layers.Dense(
        units,
        activation=activation,
        kernel_regularizer=regularizer,
        kernel_constraint=(
            tf.keras.constraints.max_norm(max_norm)
            if max_norm is not None
            else None
        ),
        bias_constraint=(
            tf.keras.constraints.max_norm(max_norm)
            if max_norm is not None
            else None
        ),
    )
