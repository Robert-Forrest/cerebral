"""Module providing model construction functionality."""

import datetime
import os
from numbers import Number
from typing import List

import pandas as pd
import numpy as np
import tensorflow as tf

import cerebral as cb

tf.keras.backend.set_floatx("float64")


def setup_losses_and_metrics():
    """Returns losses and metrics per target feature, accounting for numerical
    or categorical targets.

    :group: models
    """

    losses = {}
    feature_metrics = {}

    for feature in cb.conf.targets:
        if feature.type == "numerical":
            feature_metrics[feature.name] = [
                cb.loss.masked_MSE,
                cb.loss.masked_MAE,
                cb.loss.masked_Huber,
                cb.loss.masked_PseudoHuber,
            ]
            if feature.loss == "MSE":
                losses[feature.name] = cb.loss.masked_MSE
            elif feature.loss == "MAE":
                losses[feature.name] = cb.loss.masked_MAE
            elif feature.loss == "Huber":
                losses[feature.name] = cb.loss.masked_Huber
            elif feature.loss == "PseudoHuber":
                losses[feature.name] = cb.loss.masked_PseudoHuber
        else:
            feature_metrics[feature.name] = [
                cb.metrics.accuracy,
                cb.metrics.truePositiveRate,
                cb.metrics.trueNegativeRate,
                cb.metrics.positivePredictiveValue,
                cb.metrics.negativePredictiveValue,
                cb.metrics.balancedAccuracy,
                cb.metrics.f1,
                cb.metrics.informedness,
                cb.metrics.markedness,
            ]
            losses[
                feature.name
            ] = cb.loss.masked_sparse_categorical_crossentropy

    if len(losses) != len(cb.conf.targets):
        raise Exception("Number of losses does not match number of targets!")

    return losses, feature_metrics


def build_input_layers(train_ds) -> list:
    """Returns a list of input layers, one for each input feature.

    :group: models
    """

    return [
        tf.keras.Input(shape=(1,), name=feature)
        for feature in train_ds.element_spec[0]
    ]


def build_base_model(
    inputs,
    num_shared_layers,
    regularizer,
    regularizer_rate,
    max_norm,
    dropout,
    activation,
    units_per_layer,
):
    """Constructs the base model, shared by all feature branches. See Figure 1
    of https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: models
    """

    baseModel = None
    for i in range(num_shared_layers):
        if i == 0:
            baseModel = cb.layers.dense(
                units_per_layer,
                activation,
                regularizer,
                regularizer_rate,
                max_norm,
            )(inputs)
        else:
            baseModel = cb.layers.dense(
                units_per_layer,
                activation,
                regularizer,
                regularizer_rate,
                max_norm,
            )(baseModel)
        if dropout > 0:
            if i < num_shared_layers - 1:
                baseModel = tf.keras.layers.Dropout(dropout)(baseModel)

    return baseModel


def build_feature_branch(
    feature,
    ensemble_size,
    num_layers,
    units_layer,
    max_norm,
    activation,
    regularizer,
    regularizer_rate,
    dropout,
    inputs,
):
    """Constructs a branch of the model for a specific feature. See Figure 1
    of https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: models
    """

    ensemble = []
    for m in range(ensemble_size):
        x = None
        for i in range(num_layers):
            if i == 0:
                x = cb.layers.dense(
                    units_layer,
                    activation,
                    regularizer,
                    regularizer_rate,
                    max_norm,
                )(inputs)
            else:
                x = cb.layers.dense(
                    units_layer,
                    activation,
                    regularizer,
                    regularizer_rate,
                    max_norm,
                )(x)
            if dropout > 0:
                if i < num_layers - 1:
                    x = tf.keras.layers.Dropout(dropout)(x)
                else:
                    x = tf.keras.layers.Dropout(dropout / 5)(x)

        if feature.type == "categorical":
            if ensemble_size > 1:
                ensemble.append(
                    tf.keras.layers.Dense(
                        3,
                        activation="softmax",
                        name=feature.name + "_" + str(m),
                    )(x)
                )
            else:
                ensemble.append(
                    tf.keras.layers.Dense(
                        3, activation="softmax", name=feature.name
                    )(x)
                )
        else:
            if ensemble_size > 1:
                ensemble.append(
                    tf.keras.layers.Dense(
                        1,
                        activation="softplus",
                        name=feature.name + "_" + str(m),
                    )(x)
                )
            else:
                ensemble.append(
                    tf.keras.layers.Dense(
                        1, activation="softplus", name=feature.name
                    )(x)
                )
    return ensemble


def build_model(
    train_ds,
    train_features,
    num_shared_layers,
    num_specific_layers,
    units_per_layer,
    regularizer="l2",
    regularizer_rate=0.001,
    dropout=0.3,
    learning_rate=0.01,
    ensemble_size=1,
    activation="elu",
    max_norm=3,
):
    """Constructs a model containing a shared branch, and individual feature
    branches. See Figure 1
    of https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: models
    """

    inputs = build_input_layers(train_ds)

    normalized_inputs = []
    for input_layer in inputs:
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(
            axis=None
        )
        normalizer.adapt(train_features[input_layer.name])
        normalized_inputs.append(normalizer(input_layer))

    concatenated_inputs = tf.keras.layers.concatenate(
        normalized_inputs, name="Inputs"
    )

    if num_shared_layers > 0 and len(cb.conf.targets) > 1:
        base_model = build_base_model(
            concatenated_inputs,
            num_shared_layers,
            regularizer,
            regularizer_rate,
            max_norm,
            dropout,
            activation,
            units_per_layer,
        )

    else:
        base_model = concatenated_inputs

    # base_model = tf.keras.layers.LayerNormalization()(base_model)

    losses, metrics = setup_losses_and_metrics()
    outputs = []
    normalized_outputs = []

    for feature in cb.conf.targets:

        if len(outputs) > 0:
            model_branch = tf.keras.layers.concatenate(
                [base_model] + normalized_outputs
            )
        else:
            model_branch = base_model

        ensemble = build_feature_branch(
            feature,
            ensemble_size,
            num_specific_layers,
            units_per_layer,
            max_norm,
            activation,
            regularizer,
            regularizer_rate,
            dropout,
            model_branch,
        )

        if len(ensemble) > 1:
            outputs.append(
                tf.keras.layers.average(ensemble, name=feature.name)
            )
        else:
            outputs.append(ensemble[0])

        normalized_outputs.append(
            tf.keras.layers.LayerNormalization()(outputs[-1])
        )

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=losses,
        metrics=metrics,
        weighted_metrics=[],
        loss_weights={
            target["name"]: target["weight"] for target in cb.conf.targets
        },
        optimizer=optimiser,
    )

    return model


def save(model, path):
    """Save a model to disk.

    :group: models"""

    model.save(path)


def load(path):
    """Load a model from disk, associate all custom objects with their cerebral functions.

    :group: models"""

    return tf.keras.models.load_model(
        path,
        custom_objects={
            "accuracy": cb.metrics.accuracy,
            "truePositiveRate": cb.metrics.truePositiveRate,
            "trueNegativeRate": cb.metrics.trueNegativeRate,
            "positivePredictiveValue": cb.metrics.positivePredictiveValue,
            "negativePredictiveValue": cb.metrics.negativePredictiveValue,
            "balancedAccuracy": cb.metrics.balancedAccuracy,
            "f1": cb.metrics.f1,
            "informedness": cb.metrics.informedness,
            "markedness": cb.metrics.markedness,
            "masked_MSE": cb.loss.masked_MSE,
            "masked_MAE": cb.loss.masked_MAE,
            "masked_Huber": cb.loss.masked_Huber,
            "masked_PseudoHuber": cb.loss.masked_PseudoHuber,
            "masked_sparse_categorical_crossentropy": cb.loss.masked_sparse_categorical_crossentropy,
        },
    )


def train_model(data, max_epochs=1000):
    train, test = cb.features.train_test_split(data)

    train_compositions = train.pop("composition")
    test_compositions = test.pop("composition")

    (
        train_ds,
        test_ds,
        train_features,
        test_features,
        train_labels,
        test_labels,
    ) = cb.features.create_datasets(data, cb.conf.targets, train, test)

    train_data = {
        "compositions": train_compositions,
        "dataset": train_ds,
        "labels": train_labels,
    }

    test_data = {
        "compositions": test_compositions,
        "dataset": test_ds,
        "labels": test_labels,
    }

    model, history = compile_and_fit(
        train_ds,
        train_features,
        test_ds,
        max_epochs,
    )

    return model, history, train_data, test_data


def compile_and_fit(train_ds, train_features, test_ds=None, max_epochs=1000):
    """Compile a model, and perform training.

    :group: models
    """

    model = build_model(
        train_ds,
        train_features,
        num_shared_layers=3,
        num_specific_layers=5,
        units_per_layer=64,
        regularizer="l2",
        regularizer_rate=0.001,
        dropout=0.1,
        learning_rate=1e-2,
        activation="elu",
        max_norm=3.0,
        ensemble_size=1,
    )

    model, history = fit(
        model,
        train_ds,
        test_ds,
        max_epochs,
    )

    if cb.conf.plot:
        if cb.conf.save:
            tf.keras.utils.plot_model(
                model,
                to_file=cb.conf.image_directory + "model.png",
                rankdir="LR",
            )
        else:
            tf.keras.utils.plot_model(model, rankdir="LR")

        cb.plots.plot_training(history)

    if cb.conf.save:
        save(model, cb.conf.output_directory + "/model")

    return model, history


def fit(
    model,
    train_ds,
    test_ds,
    max_epochs=1000,
):
    """Perform training of a model to data.

    :group: models
    """

    patience = max(max_epochs // 4, 1)
    min_delta = 0.001

    reduce_lr_patience = patience // 3
    early_stop_patience = patience

    monitor = "loss"
    if test_ds is not None:
        monitor = "val_loss"

    history = model.fit(
        train_ds,
        batch_size=cb.conf.train.get("batch_size", 1024),
        epochs=max_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stop_patience,
                min_delta=min_delta,
                mode="auto",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=reduce_lr_patience,
                mode="auto",
                min_delta=min_delta * 10,
                cooldown=reduce_lr_patience // 2,
            ),
        ],
        validation_data=test_ds,
        verbose=2,
    )
    return model, history


def extract_predictions(predictions, prediction_names):

    if len(prediction_names) == 1:
        if (
            cb.conf.targets[
                cb.conf.target_names.index(prediction_names[0])
            ].type
            == "numerical"
        ):
            predictions = [predictions.flatten()]
    else:
        for i in range(len(predictions)):
            if (
                cb.conf.targets[
                    cb.conf.target_names.index(prediction_names[i])
                ].type
                == "numerical"
            ):
                predictions[i] = predictions[i].flatten()

    return predictions


def calculate_prediction_errors(
    labels: pd.DataFrame,
    predictions: List[List[Number]],
    prediction_names: List[str],
) -> List[Number]:
    """Calculate the prediction errors for each labelled item of training data.

    :group: models

    Parameters
    ----------

    labels
        True numerical values.
    predictions
        Predicted numerical values.
    prediction_names
        List of names of each predicted feature.

    """

    errors = []
    for i in range(len(predictions)):
        errors.append([])
        target = cb.conf.targets[
            cb.conf.target_names.index(prediction_names[i])
        ]

        if target["type"] == "numerical":
            for j in range(len(predictions[i])):
                if labels[target["name"]].iloc[j] != cb.features.mask_value:
                    errors[i].append(
                        predictions[i][j] - labels[target["name"]].iloc[j]
                    )
                else:
                    errors[i].append(0)
        else:
            for j in range(len(predictions[i])):
                if (
                    labels[target["name"]].iloc[j] != cb.features.mask_value
                ) and (
                    np.argmax(predictions[i][j])
                    != labels[target["name"]].iloc[j]
                ):
                    errors[i].append(True)
                else:
                    errors[i].append(False)

    return errors


def calculate_regression_metrics(
    labels: List[Number], predictions: List[Number]
) -> dict:
    """Calculate regression metrics to evaluate model performance.

    :group: models

    Parameters
    ----------

    labels
        True numerical values.
    predictions
        Predicted numerical values.

    """

    metrics = {}
    metrics["R_sq"] = cb.metrics.calc_R_sq(labels, predictions)
    metrics["RMSE"] = cb.metrics.calc_RMSE(labels, predictions)
    metrics["MAE"] = cb.metrics.calc_MAE(labels, predictions)
    return metrics


def evaluate_model(
    model,
    train_ds,
    train_labels,
    test_ds=None,
    test_labels=None,
    train_compositions=None,
    test_compositions=None,
):
    """Evaluate the performance of a trained model by comparison to known data.

    :group: models
    """

    metrics = {}
    train_errorbars = None
    test_errorbars = None

    train_predictions = []

    masked_train_labels = {}
    masked_train_predictions = {}

    prediction_names = [
        f["name"] for f in get_model_prediction_features(model)
    ]

    train_predictions = extract_predictions(
        model.predict(train_ds), prediction_names
    )
    train_errors = calculate_prediction_errors(
        train_labels, train_predictions, prediction_names
    )

    for i, target_name in enumerate(prediction_names):
        (
            masked_train_labels[target_name],
            masked_train_predictions[target_name],
        ) = cb.features.filter_masked(
            train_labels[target_name], train_predictions[i]
        )
        metrics[target_name] = {}
        metrics[target_name]["train"] = calculate_regression_metrics(
            masked_train_labels[target_name],
            masked_train_predictions[target_name],
        )

    test_predictions = None
    test_errors = None
    masked_test_labels = None
    masked_test_predictions = None
    if test_ds:
        masked_test_labels = {}
        masked_test_predictions = {}

        test_predictions = extract_predictions(
            model.predict(test_ds), prediction_names
        )
        test_errors = calculate_prediction_errors(
            test_labels, test_predictions, prediction_names
        )
        for i, target_name in enumerate(prediction_names):
            (
                masked_test_labels[target_name],
                masked_test_predictions[target_name],
            ) = cb.features.filter_masked(
                test_labels[target_name], test_predictions[i]
            )
            metrics[target_name]["test"] = calculate_regression_metrics(
                masked_test_labels[target_name],
                masked_test_predictions[target_name],
            )

    if cb.conf.save:
        cb.plots.write_errors(
            train_compositions, train_labels, train_predictions
        )
        if test_ds is not None:
            cb.plots.write_errors(
                test_compositions, test_labels, test_predictions, suffix="test"
            )

    if cb.conf.plot:

        cb.plots.plot_results_classification(
            train_labels, train_predictions, test_labels, test_predictions
        )
        cb.plots.plot_results_regression(
            train_labels,
            train_predictions,
            train_errors,
            test_labels,
            test_predictions,
            test_errors,
            metrics=metrics,
            train_compositions=train_compositions,
            test_compositions=test_compositions,
            train_errorbars=train_errorbars,
            test_errorbars=test_errorbars,
        )

    if test_ds is not None:
        return (
            train_predictions,
            train_errors,
            test_predictions,
            test_errors,
            metrics,
        )
    else:
        return train_predictions, train_errors, metrics


def get_model_prediction_features(model):
    """Extract the names and types of output features from a model.

    :group: models
    """

    predictions = []
    for node in model.outputs:
        split_name = node.name.split("/")

        if "Softmax" in split_name[1]:
            dtype = "categorical"
        else:
            dtype = "numerical"

        predictions.append({"name": split_name[0], "type": dtype})

    return predictions


def get_model_input_features(model):
    """Extract the names of expected input features from a model.

    :group: models
    """

    inputs = []
    for node in model.inputs:
        inputs.append(node.name.split("/")[0])
    return inputs


def predict(model, alloys):
    """Use a trained model to produce predictions for a set of alloy
    compositions.

    :group: models
    """

    prediction_features = get_model_prediction_features(model)

    data = cb.features.calculate_features(
        alloys,
        model=model,
        merge_duplicates=False,
        drop_correlated_features=False,
    )
    compositions = data.pop("composition")

    raw_predictions = model.predict(
        cb.features.df_to_dataset(data, prediction_features)
    )

    predictions = {}
    for i in range(len(raw_predictions)):
        predictions[prediction_features[i]["name"]] = raw_predictions[i]

        if prediction_features[i]["type"] == "numerical":
            predictions[prediction_features[i]["name"]] = predictions[
                prediction_features[i]["name"]
            ].flatten()

    return predictions
