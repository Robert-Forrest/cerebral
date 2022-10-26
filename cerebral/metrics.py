"""Module providing metric calculation functionality."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

import cerebral as cb

K = tf.keras.backend


def tprPerClass(y_true, y_pred, class_index=0):
    """Calculate the per-class true positive rate, ignoring masked values.

    :group: metrics
    """

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, cb.features.mask_value), "int64")

    pp = K.cast(K.equal(pred, class_index), "int64") * mask
    p = K.cast(K.equal(true, class_index), "int64")
    tp = K.dot(K.reshape(pp, (1, -1)), K.reshape(p, (-1, 1)))

    return K.cast(K.sum(tp), "float64") / (
        K.cast(K.sum(p), "float64") + K.epsilon()
    )


def truePositiveRate(y_true, y_pred):
    """Calculate the true positive rate, ignoring masked values.

    :group: metrics
    """

    return (
        tprPerClass(y_true, y_pred, 0)
        + tprPerClass(y_true, y_pred, 1)
        + tprPerClass(y_true, y_pred, 2)
    ) / 3


def ppvPerClass(y_true, y_pred, class_index=0):
    """Calculate the per-class positive predictive value, ignoring masked values.

    :group: metrics
    """

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, cb.features.mask_value), "int64")

    pp = K.cast(K.equal(pred, class_index), "int64") * mask
    p = K.cast(K.equal(true, class_index), "int64")
    tp = K.dot(K.reshape(pp, (1, -1)), K.reshape(p, (-1, 1)))

    return K.cast(K.sum(tp), "float64") / (
        K.cast(K.sum(pp), "float64") + K.epsilon()
    )


def positivePredictiveValue(y_true, y_pred):
    """Calculate the positive predictive value, ignoring masked values.

    :group: metrics
    """

    return (
        ppvPerClass(y_true, y_pred, 0)
        + ppvPerClass(y_true, y_pred, 1)
        + ppvPerClass(y_true, y_pred, 2)
    ) / 3


def f1(y_true, y_pred):
    """Calculate the f1 score, ignoring masked values.

    :group: metrics
    """

    positivePredictiveValue_val = positivePredictiveValue(y_true, y_pred)
    truePositiveRate_val = truePositiveRate(y_true, y_pred)
    return (2 * positivePredictiveValue_val * truePositiveRate_val) / (
        positivePredictiveValue_val
        + truePositiveRate_val
        + tf.keras.backend.epsilon()
    )


def tnrPerClass(y_true, y_pred, class_index=0):
    """Calculate the per-class true negative rate, ignoring masked values.

    :group: metrics
    """

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, cb.features.mask_value), "int64")

    pn = K.cast(K.not_equal(pred, class_index), "int64") * mask
    n = K.cast(K.not_equal(true, class_index), "int64")
    tn = K.dot(K.reshape(pn, (1, -1)), K.reshape(n, (-1, 1)))

    return K.cast(K.sum(tn), "float64") / (
        K.cast(K.sum(n), "float64") + K.epsilon()
    )


def trueNegativeRate(y_true, y_pred):
    """Calculate the overall true negative rate, ignoring masked values.

    :group: metrics
    """

    return (
        tnrPerClass(y_true, y_pred, 0)
        + tnrPerClass(y_true, y_pred, 1)
        + tnrPerClass(y_true, y_pred, 2)
    ) / 3


def npvPerClass(y_true, y_pred, class_index=0):
    """Calculate the per-class negative predictive value, ignoring masked values.

    :group: metrics
    """

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, -1), "int64")

    pn = K.cast(K.not_equal(pred, class_index), "int64") * mask
    n = K.cast(K.not_equal(true, class_index), "int64")
    tn = K.dot(K.reshape(pn, (1, -1)), K.reshape(n, (-1, 1)))

    return K.cast(K.sum(tn), "float64") / (
        K.cast(K.sum(pn), "float64") + K.epsilon()
    )


def negativePredictiveValue(y_true, y_pred):
    """Calculate the overall negative predictive value, ignoring masked values.

    :group: metrics
    """

    return (
        npvPerClass(y_true, y_pred, 0)
        + npvPerClass(y_true, y_pred, 1)
        + npvPerClass(y_true, y_pred, 2)
    ) / 3


def informedness(y_true, y_pred):
    """Calculate the informedness, ignoring masked values.

    :group: metrics
    """

    return (
        truePositiveRate(y_true, y_pred) + trueNegativeRate(y_true, y_pred) - 1
    )


def markedness(y_true, y_pred):
    """Calculate the markedness, ignoring masked values.

    :group: metrics
    """

    return (
        positivePredictiveValue(y_true, y_pred)
        + negativePredictiveValue(y_true, y_pred)
        - 1
    )


def accuracy(y_true, y_pred):
    """Calculate the accuracy, ignoring masked values.

    :group: metrics
    """

    pred = K.argmax(y_pred)
    true = K.cast(K.squeeze(y_true, axis=-1), "int64")

    mask = K.cast(K.not_equal(true, cb.features.mask_value), "int64")
    matches = K.cast(K.equal(true, pred), "int64") * mask

    return K.sum(matches) / K.maximum(K.sum(mask), 1)


def balancedAccuracy(y_true, y_pred) -> float:
    """Calculate the balanced accuracy, ignoring masked values.

    :group: metrics
    """

    return 0.5 * (
        truePositiveRate(y_true, y_pred) + trueNegativeRate(y_true, y_pred)
    )


def calc_R_sq(true, prediction) -> float:
    """Calculate the R-squared score.

    :group: metrics
    """

    true_mean = np.mean(true)

    SS_tot = 0

    for y in true:
        SS_tot += (y - true_mean) ** 2

    SS_res = 0
    for (y, z) in zip(true, prediction):
        SS_res += (y - z) ** 2

    return 1 - (SS_res / SS_tot)


def calc_RMSE(true, prediction) -> float:
    """Interface to calculate the root-mean-squared-error, ignoring masked values.

    :group: metrics
    """

    return np.sqrt(mean_squared_error(true, prediction))


def calc_MAE(true, prediction) -> float:
    """Interface to calculate the mean-absolute-error, ignoring masked values.

    :group: metrics
    """

    return mean_absolute_error(true, prediction)


def calc_accuracy(true, prediction) -> float:
    """Interface to calculate the accuracy.

    :group: metrics
    """

    if isinstance(prediction[0], (list, tuple, np.ndarray)):
        predictionMax = []
        for p in prediction:
            predictionMax.append(np.argmax(p))

        return accuracy_score(true, predictionMax)
    else:
        return accuracy_score(true, prediction)


def calc_f1(true, prediction) -> float:
    """Interface to calculate the F1 score.

    :group: metrics
    """

    if isinstance(prediction[0], (list, tuple, np.ndarray)):
        predictionMax = []
        for p in prediction:
            predictionMax.append(np.argmax(p))

        return f1_score(true, predictionMax, average="macro")
    else:
        return f1_score(true, prediction, average="macro")


def calc_recall(true, prediction) -> float:
    """Interface to calculate the recall score.

    :group: metrics
    """

    if isinstance(prediction[0], (list, tuple, np.ndarray)):
        predictionMax = []
        for p in prediction:
            predictionMax.append(np.argmax(p))

        return recall_score(true, predictionMax, average="macro")
    else:
        return recall_score(true, prediction, average="macro")


def calc_precision(true, prediction) -> float:
    """Interface to calculate the precision score.

    :group: metrics
    """

    if isinstance(prediction[0], (list, tuple, np.ndarray)):
        predictionMax = []
        for p in prediction:
            predictionMax.append(np.argmax(p))

        return precision_score(true, predictionMax, average="macro")
    else:
        return precision_score(true, prediction, average="macro")


def meanAbsoluteDeviation(data) -> float:
    """Calculate the mean-absolute-deviation of a dataset.

    :group: metrics
    """

    mean = np.mean(data)

    mad = 0
    for datum in data:
        mad += np.abs(datum - mean)
    mad /= len(data)
    return mad


def rootMeanSquareDeviation(data) -> float:
    """Calculate the root-mean-square-deviation of a dataset.

    :group: metrics
    """

    mean = np.mean(data)

    RMSD = 0
    for datum in data:
        RMSD += (datum - mean) ** 2
    RMSD /= len(data)

    return np.sqrt(RMSD)
