import tensorflow as tf


def dense(units, activation, regularizer,
          regularizer_rate, max_norm):

    if regularizer == 'l1':
        regularizer = tf.keras.regularizers.l1(regularizer_rate)
    elif regularizer == 'l2':
        regularizer = tf.keras.regularizers.l2(regularizer_rate)
    elif regularizer == 'l1l2':
        regularizer = tf.keras.regularizers.L1L2(regularizer_rate)

    return tf.keras.layers.Dense(units, activation=activation,
                                 # activity_regularizer=regularizer,
                                 kernel_regularizer=regularizer,
                                 # bias_regularizer=regularizer,
                                 kernel_constraint=tf.keras.constraints.max_norm(
                                     max_norm),
                                 bias_constraint=tf.keras.constraints.max_norm(max_norm))
