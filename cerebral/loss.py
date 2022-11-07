"""Module providing loss-related functionality."""

import tensorflow as tf

import cerebral as cb


K = tf.keras.backend


def masked_MSE(y_true, y_pred):
    """Calculate the mean-squared-error, ignoring masked values.

    :group: loss
    """

    mask = K.not_equal(y_true, cb.features.mask_value)

    squared_error = tf.where(
        mask, tf.math.square(tf.subtract(y_true, y_pred)), 0
    )

    return squared_error


def masked_MAE(y_true, y_pred):
    """Calculate the mean-absolute-error, ignoring masked values.

    :group: loss
    """

    mask = K.not_equal(y_true, cb.features.mask_value)

    abs_error = tf.where(mask, tf.math.abs(tf.subtract(y_true, y_pred)), 0)

    return abs_error


def masked_PseudoHuber(y_true, y_pred):
    """Calculate the pseudo-Huber error, ignoring masked values.

    :group: loss
    """

    mask = K.not_equal(y_true, cb.features.mask_value)

    error = tf.where(mask, tf.subtract(y_true, y_pred), 0)

    huber = tf.math.subtract(
        tf.math.sqrt(
            tf.math.add(K.cast(1.0, "float64"), tf.math.square(error))
        ),
        K.cast(1.0, "float64"),
    )

    return huber


def masked_Huber(y_true, y_pred):
    """Calculate the Huber error, ignoring masked values.

    :group: loss
    """

    mask = K.not_equal(y_true, cb.features.mask_value)

    error = tf.where(mask, tf.abs(tf.subtract(y_true, y_pred)), 0)

    delta = K.cast(1.0, "float64")

    huber = tf.where(
        tf.abs(error) > delta,
        tf.add(
            K.cast(0.5 * delta**2, "float64"),
            tf.multiply(delta, tf.subtract(error, delta)),
        ),
        tf.multiply(K.cast(0.5, "float64"), tf.square(error)),
    )

    return huber


def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Calculate the sparse categorical cross-entropy, ignoring masked
    values.

    :group: loss
    """

    mask = K.not_equal(K.squeeze(y_true, axis=-1), cb.features.mask_value)

    scce = tf.where(
        mask,
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred),
        0,
    )

    return scce
