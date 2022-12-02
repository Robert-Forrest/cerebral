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
                cb.metrics.matthewsCorrelation,
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

    inputs = []
    for feature in train_ds.element_spec[0]:
        if feature != "composition":
            inputs.append(tf.keras.Input(shape=(1,), name=feature))
    return inputs


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
    num_shared_layers,
    num_specific_layers,
    units_per_layer,
    regularizer="l2",
    regularizer_rate=0.001,
    dropout=0.3,
    learning_rate=0.01,
    ensemble_size=1,
    activation="relu",
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
        normalizer = tf.keras.layers.Normalization(axis=None)
        normalizer.adapt(train_ds.map(lambda x, y, z: x[input_layer.name]))
        normalized_inputs.append(normalizer(input_layer))

    concatenated_inputs = tf.keras.layers.concatenate(
        normalized_inputs, name="Inputs"
    )

    if num_shared_layers > 0:
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
            "matthewsCorrelation": cb.metrics.matthewsCorrelation,
            "masked_MSE": cb.loss.masked_MSE,
            "masked_MAE": cb.loss.masked_MAE,
            "masked_Huber": cb.loss.masked_Huber,
            "masked_PseudoHuber": cb.loss.masked_PseudoHuber,
            "masked_sparse_categorical_crossentropy": cb.loss.masked_sparse_categorical_crossentropy,
        },
    )


def train_model(data, max_epochs=1000, early_stop=True):

    max_epochs = cb.conf.train.get("max_epochs", max_epochs)

    if cb.conf.train.train_percentage < 1.0:
        train, test = cb.features.train_test_split(
            data, train_percentage=cb.conf.train.train_percentage
        )

        (
            train_ds,
            test_ds,
        ) = cb.features.create_datasets(data, cb.conf.targets, train, test)

        model, history = compile_and_fit(
            train_ds,
            test_ds=test_ds,
            max_epochs=max_epochs,
            early_stop=early_stop,
        )

        return model, history, train_ds, test_ds

    else:

        train_ds = cb.features.create_datasets(
            data, cb.conf.targets, train=data
        )

        model, history = compile_and_fit(
            train_ds, max_epochs=max_epochs, early_stop=early_stop
        )

        return model, history, train_ds


def compile_and_fit(train_ds, test_ds=None, max_epochs=1000, early_stop=True):
    """Compile a model, and perform training.

    :group: models
    """

    max_epochs = cb.conf.train.get("max_epochs", max_epochs)

    model = build_model(
        train_ds,
        num_shared_layers=cb.conf.train.get("num_shared_layers", 3),
        num_specific_layers=cb.conf.train.get("num_specific_layers", 5),
        units_per_layer=cb.conf.train.get("units_per_layer", 64),
        regularizer=cb.conf.train.get("regularizer", "l2"),
        regularizer_rate=cb.conf.train.get("regularizer_rate", 0.0001),
        dropout=cb.conf.train.get("dropout", 0.3),
        learning_rate=cb.conf.train.get("learning_rate", 0.01),
        activation=cb.conf.train.get("activation", "relu"),
        max_norm=cb.conf.train.get("max_norm", 3.0),
        ensemble_size=1,
    )

    if cb.conf.plot.model:
        if cb.conf.save:
            model_plot_path = cb.conf.output_directory
            if cb.conf.model_name not in model_plot_path:
                model_plot_path += cb.conf.model_name + "/"
            if not os.path.exists(model_plot_path):
                os.makedirs(model_plot_path, exist_ok=True)

            tf.keras.utils.plot_model(
                model,
                to_file=model_plot_path + cb.conf.model_name + ".png",
                rankdir="LR",
            )
        else:
            tf.keras.utils.plot_model(model, rankdir="LR")

    if test_ds is None:
        model, history = fit(
            model, train_ds, max_epochs=max_epochs, early_stop=early_stop
        )
    else:
        model, history = fit(
            model,
            train_ds,
            test_ds=test_ds,
            max_epochs=max_epochs,
            early_stop=early_stop,
        )

    if cb.conf.plot.model:
        cb.plots.plot_training(history)

    if cb.conf.save:
        model_path = cb.conf.output_directory
        if cb.conf.model_name not in model_path:
            model_path += "/" + cb.conf.model_name
        save(model, model_path)

    return model, history


def fit(model, train_ds, test_ds=None, max_epochs=1000, early_stop=True):
    """Perform training of a model to data.

    :group: models
    """

    patience = min(500, max(max_epochs // 3, 1))
    min_delta = 0.001

    reduce_lr_patience = patience // 3
    early_stop_patience = patience

    monitor = "loss"
    if test_ds is not None:
        monitor = "val_loss"

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=reduce_lr_patience,
            mode="auto",
            min_delta=min_delta * 10,
            cooldown=reduce_lr_patience // 2,
        ),
    ]
    if early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stop_patience,
                min_delta=min_delta,
                mode="auto",
                restore_best_weights=True,
            )
        )

    history = model.fit(
        train_ds,
        batch_size=cb.conf.train.get("batch_size", 1024),
        epochs=max_epochs,
        callbacks=callbacks,
        validation_data=test_ds,
        verbose=2,
    )
    return model, history


def extract_predictions_truths(model, dataset, prediction_names):

    predictions = {t: [] for t in prediction_names}
    truths = {t: [] for t in prediction_names}
    compositions = []

    for example, true, weight in dataset:
        compositions.extend(example["composition"].numpy())

        p = model.predict(example)

        for i in range(len(predictions)):
            if (
                cb.conf.targets[
                    cb.conf.target_names.index(prediction_names[i])
                ].type
                == "numerical"
            ):
                if len(predictions) == 1:
                    predictions[prediction_names[i]].extend(p.flatten())
                else:
                    predictions[prediction_names[i]].extend(p[i].flatten())
            else:
                if len(predictions) == 1:
                    predictions[prediction_names[i]].extend(p)
                else:
                    predictions[prediction_names[i]].extend(p[i])

            truths[prediction_names[i]].extend(
                true[prediction_names[i]].numpy()
            )

    for i in range(len(compositions)):
        compositions[i] = compositions[i].decode("UTF-8")

    return predictions, truths, compositions


def calculate_prediction_errors(
    truth: dict,
    predictions: dict,
    prediction_names: List[str],
) -> dict:
    """Calculate the prediction errors for each item of training data.

    :group: models

    Parameters
    ----------

    truth
        True numerical values.
    predictions
        Predicted numerical values.
    prediction_names
        List of names of each predicted feature.

    """

    errors = {}
    for feature in predictions:
        errors[feature] = []
        target = cb.conf.targets[cb.conf.target_names.index(feature)]

        if target["type"] == "numerical":
            for j in range(len(predictions[feature])):
                if truth[feature][j] != cb.features.mask_value:
                    errors[feature].append(
                        predictions[feature][j] - truth[feature][j]
                    )
                else:
                    errors[feature].append(0)
        else:
            for j in range(len(predictions[feature])):
                if (truth[feature][j] != cb.features.mask_value) and (
                    np.argmax(predictions[feature][j])
                    != truth[target["name"]][j]
                ):
                    errors[feature].append(True)
                else:
                    errors[feature].append(False)

    return errors


def calculate_regression_metrics(
    truth: List[Number], predictions: List[Number]
) -> dict:
    """Calculate regression metrics to evaluate model performance.

    :group: models

    Parameters
    ----------

    truth
        True numerical values.
    predictions
        Predicted numerical values.

    """

    metrics = {}
    metrics["R_sq"] = cb.metrics.calc_R_sq(truth, predictions)
    metrics["RMSE"] = cb.metrics.calc_RMSE(truth, predictions)
    metrics["MAE"] = cb.metrics.calc_MAE(truth, predictions)
    return metrics


def calculate_classification_metrics(
    truth: List[Number], predictions: List[Number]
) -> dict:
    """Calculate classification metrics to evaluate model performance.

    :group: models

    Parameters
    ----------

    truth
        True numerical values.
    predictions
        Predicted numerical values.

    """

    metrics = {}
    metrics["accuracy"] = cb.metrics.calc_accuracy(truth, predictions)
    metrics["recall"] = cb.metrics.calc_recall(truth, predictions)
    metrics["precision"] = cb.metrics.calc_precision(truth, predictions)
    metrics["trueNegativeRate"] = cb.metrics.calc_trueNegativeRate(
        truth, predictions
    ).numpy()
    metrics["f1"] = cb.metrics.calc_f1(truth, predictions)
    metrics["informedness"] = cb.metrics.informedness(
        truth, predictions
    ).numpy()
    metrics["markedness"] = cb.metrics.markedness(truth, predictions).numpy()
    metrics["matthewsCorrelation"] = cb.metrics.matthewsCorrelation(
        truth, predictions
    ).numpy()

    return metrics


def evaluate_model(model, train_ds, test_ds=None):
    """Evaluate the performance of a trained model by comparison to known data.

    :group: models
    """

    metrics = {}

    train_predictions = []

    masked_train_truth = {}
    masked_train_predictions = {}

    prediction_names = [
        f["name"] for f in get_model_prediction_features(model)
    ]

    (
        train_predictions,
        train_truth,
        train_compositions,
    ) = extract_predictions_truths(model, train_ds, prediction_names)

    train_errors = calculate_prediction_errors(
        train_truth, train_predictions, prediction_names
    )

    for i, target_name in enumerate(prediction_names):
        target = None
        for t in cb.conf.targets:
            if t["name"] == target_name:
                target = t
                break

        (
            masked_train_truth[target_name],
            masked_train_predictions[target_name],
        ) = cb.features.filter_masked(
            train_truth[target_name], train_predictions[target_name]
        )

        metrics[target_name] = {}
        if target["type"] == "numerical":
            metrics[target_name]["train"] = calculate_regression_metrics(
                masked_train_truth[target_name],
                masked_train_predictions[target_name],
            )
        else:
            metrics[target_name]["train"] = calculate_classification_metrics(
                masked_train_truth[target_name],
                masked_train_predictions[target_name],
            )

    test_predictions = None
    test_truth = None
    test_errors = None
    test_compositions = None

    if test_ds:
        masked_test_truth = {}
        masked_test_predictions = {}

        (
            test_predictions,
            test_truth,
            test_compositions,
        ) = extract_predictions_truths(model, test_ds, prediction_names)

        test_errors = calculate_prediction_errors(
            test_truth, test_predictions, prediction_names
        )

        for i, target_name in enumerate(prediction_names):
            target = None
            for t in cb.conf.targets:
                if t["name"] == target_name:
                    target = t
                    break

            (
                masked_test_truth[target_name],
                masked_test_predictions[target_name],
            ) = cb.features.filter_masked(
                test_truth[target_name], test_predictions[target_name]
            )

            if target["type"] == "numerical":
                metrics[target_name]["test"] = calculate_regression_metrics(
                    masked_test_truth[target_name],
                    masked_test_predictions[target_name],
                )
            else:
                metrics[target_name][
                    "test"
                ] = calculate_classification_metrics(
                    masked_test_truth[target_name],
                    masked_test_predictions[target_name],
                )

    if cb.conf.save:
        cb.plots.write_errors(
            train_compositions, train_truth, train_predictions
        )
        if test_ds is not None:
            cb.plots.write_errors(
                test_compositions, test_truth, test_predictions, suffix="test"
            )

    if cb.conf.plot.model:

        cb.plots.plot_results_classification(
            train_truth,
            train_predictions,
            test_truth,
            test_predictions,
            metrics=metrics,
        )
        cb.plots.plot_results_regression(
            train_truth,
            train_predictions,
            train_errors,
            test_truth,
            test_predictions,
            test_errors,
            metrics=metrics,
            train_compositions=train_compositions,
            test_compositions=test_compositions,
        )

    if test_ds is not None:
        return (
            {
                "predictions": train_predictions,
                "truth": train_truth,
                "errors": train_errors,
            },
            {
                "predictions": test_predictions,
                "truth": test_truth,
                "errors": test_errors,
            },
            metrics,
        )
    else:
        return (
            {
                "predictions": train_predictions,
                "truth": train_truth,
                "errors": train_errors,
            },
            metrics,
        )


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


def predict(model, alloys, uncertainty=False):
    """Use a trained model to produce predictions for a set of alloy
    compositions.

    :group: models
    """

    data, targets, input_features = cb.features.calculate_features(
        alloys,
        model=model,
        merge_duplicates=False,
        drop_correlated_features=False,
    )

    data.pop("composition")
    inputs = cb.features.df_to_dataset(data, targets, shuffle=False)

    if uncertainty:
        collected_predictions = {f["name"]: [] for f in targets}
        predictions = {f["name"]: [] for f in targets}
        for i in range(100):
            current_predictions = extract_predictions_training(
                model,
                inputs,
                targets,
            )
            for feature in collected_predictions:
                collected_predictions[feature].append(
                    current_predictions[feature]
                )
        for feature in collected_predictions:
            for i in range(len(collected_predictions[feature][0])):
                alloy_predictions = []
                for j in range(len(collected_predictions[feature])):
                    alloy_predictions.append(
                        collected_predictions[feature][j][i]
                    )

                if isinstance(alloy_predictions[0], tf.Tensor):
                    for j in range(len(alloy_predictions)):
                        alloy_predictions[j] = alloy_predictions[j].numpy()

                if isinstance(alloy_predictions[0], (list, np.ndarray)):
                    per_class_predictions = []
                    for j in range(len(alloy_predictions[0])):
                        per_class_predictions.append([])
                        for i in range(len(alloy_predictions)):
                            per_class_predictions[j].append(
                                alloy_predictions[i][j]
                            )

                    average = []
                    CI = []
                    for j in range(len(alloy_predictions[0])):
                        average.append(np.mean(per_class_predictions[j]))
                        CI.append(
                            1.96
                            * np.std(per_class_predictions[j])
                            / np.sqrt(len(per_class_predictions[j]))
                        )

                    predictions[feature].append((average, CI))

                else:
                    predictions[feature].append(
                        (np.mean(alloy_predictions), np.std(alloy_predictions))
                    )

        return predictions
    else:
        return extract_predictions(
            model.predict(inputs),
            targets,
        )


def extract_predictions_training(model, dataset, prediction_features):

    prediction_names = [f["name"] for f in prediction_features]
    predictions = {t: [] for t in prediction_names}

    for example in dataset:

        p = model(example, training=True)
        if not isinstance(p, list):
            p = p.numpy()

        for i in range(len(predictions)):
            if prediction_features[i]["type"] == "numerical":
                if len(predictions) == 1:
                    predictions[prediction_names[i]].extend(p.flatten())
                else:
                    if not isinstance(i, np.ndarray):
                        p[i] = p[i].numpy()
                    predictions[prediction_names[i]].extend(p[i].flatten())
            else:
                if len(predictions) == 1:
                    predictions[prediction_names[i]].extend(p)
                else:
                    predictions[prediction_names[i]].extend(p[i])

    return predictions


def extract_predictions(raw_predictions, prediction_features):
    predictions = {}
    if len(prediction_features) > 1:
        for i in range(len(raw_predictions)):
            predictions[prediction_features[i]["name"]] = raw_predictions[i]

            if prediction_features[i]["type"] == "numerical":
                predictions[prediction_features[i]["name"]] = predictions[
                    prediction_features[i]["name"]
                ].flatten()
    else:
        predictions[prediction_features[0]["name"]] = raw_predictions.flatten()

    return predictions
