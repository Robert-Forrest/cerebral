import tensorflow as tf

import cerebral as cb
from . import metrics
from . import features

K = tf.keras.backend


def masked_MSE(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    squared_error = tf.where(
        mask, tf.math.square(tf.subtract(y_true, y_pred)), 0)

    # return tf.divide(tf.reduce_sum(squared_error), K.cast(tf.math.count_nonzero(K.cast(mask, 'int32')), 'float64'))
    return squared_error


def masked_MAE(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    abs_error = tf.where(
        mask, tf.math.abs(tf.subtract(y_true, y_pred)), 0)

    return abs_error
    # return tf.divide(tf.reduce_sum(abs_error), K.cast(tf.math.count_nonzero(K.cast(mask, 'int32')), 'float64'))


def masked_PseudoHuber(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    error = tf.where(mask, tf.subtract(y_true, y_pred), 0)

    huber = tf.math.subtract(tf.math.sqrt(
        tf.math.add(K.cast(1.0, 'float64'), tf.math.square(error))), K.cast(1.0, 'float64'))

    return huber


def masked_Huber(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    error = tf.where(mask, tf.abs(tf.subtract(y_true, y_pred)), 0)

    delta = K.cast(1.0, 'float64')

    huber = tf.where(tf.abs(error) > delta,
                     tf.add(K.cast(0.5*delta**2, 'float64'), tf.multiply(delta,
                                                                         tf.subtract(error, delta))),
                     tf.multiply(K.cast(0.5, 'float64'), tf.square(error)))

    return huber


def masked_sparse_categorical_crossentropy(y_true, y_pred):

    mask = K.not_equal(K.squeeze(y_true, axis=-1), features.maskValue)

    scce = tf.where(
        mask, tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred), 0)

    return scce


def negloglik(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)
    log_prob = -y_pred.log_prob(y_true)

    mask = tf.reshape(mask, [-1])
    nonzero = tf.math.count_nonzero(mask)

    return K.sum(tf.where(mask, log_prob, 0)) / tf.cast(nonzero, 'float64')
