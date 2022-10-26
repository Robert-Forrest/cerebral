"""Module providing model construction functionality."""

import os
import datetime

import pandas as pd
import tensorflow as tf

import cerebral as cb

tf.keras.backend.set_floatx("float64")
physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("Could not set memory growth")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel("INFO")


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

    return losses, feature_metrics


def build_input_layers(train_features: pd.DataFrame) -> list:
    """Returns a list of input layers, one for each input feature.

    :group: models
    """

    inputs = []
    for feature in train_features.columns:
        inputs.append(
            tf.keras.Input(shape=(1,), name=feature, dtype="float64")
        )

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
    train_features,
    train_labels,
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

    inputs = build_input_layers(train_features)

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
        baseModel = build_base_model(
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
        baseModel = concatenated_inputs

    # baseModel = tf.keras.layers.LayerNormalization()(baseModel)

    losses, metrics = setup_losses_and_metrics()
    outputs = []
    normalized_outputs = []

    for feature in cb.conf.targets:

        if len(outputs) > 0:
            model_branch = tf.keras.layers.concatenate(
                [baseModel] + normalized_outputs
            )
        else:
            model_branch = baseModel

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
        loss_weights={
            target["name"]: target["weight"] for target in cb.conf.targets
        },
        optimizer=optimiser,
        run_eagerly=False,
    )

    tf.keras.utils.plot_model(
        model, to_file=cb.conf.image_directory + "model.png", rankdir="LR"
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


def compile_and_fit(
    train_features,
    train_labels,
    sampleWeight,
    test_features=None,
    test_labels=None,
    sampleWeight_test=None,
    maxEpochs=1000,
):
    """Compile a model, and perform training.

    :group: models
    """

    model = build_model(
        train_features,
        train_labels,
        num_shared_layers=3,
        num_specific_layers=5,
        units_per_layer=64,
        regularizer="l2",
        regularizer_rate=0.001,
        dropout=0.1,
        learning_rate=1e-2,
        # learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(
        #     0.01,
        #     decay_steps=1,
        #     decay_rate=0.01,
        #     staircase=False),
        activation="elu",
        max_norm=5.0,
        ensemble_size=1,
    )

    return fit(
        model,
        train_features,
        train_labels,
        sampleWeight,
        test_features,
        test_labels,
        sampleWeight_test,
        maxEpochs,
    )


def fit(
    model,
    train_features,
    train_labels,
    sampleWeight,
    test_features=None,
    test_labels=None,
    sampleWeight_test=None,
    maxEpochs=1000,
):
    """Perform training of a model to data.

    :group: models
    """

    patience = 100
    min_delta = 0.001

    xTrain = {}
    for feature in train_features:
        xTrain[feature] = train_features[feature]

    yTrain = {}
    for feature in cb.conf.targets:
        if feature.name in train_labels:
            yTrain[feature.name] = train_labels[feature.name]

    monitor = "loss"

    testData = None
    if test_features is not None:
        xTest = {}
        for feature in test_features:
            xTest[feature] = test_features[feature]

        yTest = {}
        for feature in cb.conf.targets:
            if feature.name in test_labels:
                yTest[feature.name] = test_labels[feature.name]

        testData = (xTest, yTest, sampleWeight_test)

        monitor = "val_loss"

    history = model.fit(
        x=xTrain,
        y=yTrain,
        batch_size=cb.conf.train.get("batch_size", 1024),
        epochs=maxEpochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                mode="auto",
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience // 3,
                mode="auto",
                min_delta=min_delta * 10,
                cooldown=patience // 4,
                min_lr=0,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=cb.conf.output_directory
                + "/logs/fit/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                histogram_freq=1,
            ),
        ],
        sample_weight=sampleWeight,
        validation_data=testData,
        verbose=2,
    )
    return model, history


def train_model(
    train_features,
    train_labels,
    sampleWeight,
    test_features=None,
    test_labels=None,
    sampleWeight_test=None,
    plot=True,
    maxEpochs=1000,
    model_name=None,
):
    """Helper function which compiles and trains a model, and can also plot
    training metrics.

    :group: models
    """

    model, history = compile_and_fit(
        train_features,
        train_labels,
        sampleWeight,
        test_features=test_features,
        test_labels=test_labels,
        sampleWeight_test=sampleWeight_test,
        maxEpochs=maxEpochs,
    )

    if plot:
        cb.plots.plot_training(history, model_name=model_name)
        if model_name is not None:
            save(model, cb.conf.output_directory + "/model_" + str(model_name))
        else:
            save(model, cb.conf.output_directory + "/model")
    return model


def evaluate_model(
    model,
    train_ds,
    train_labels,
    test_ds=None,
    test_labels=None,
    plot=True,
    train_compositions=None,
    test_compositions=None,
    model_name=None,
):
    """Evaluate the performance of a trained model by comparison to known data.

    :group: models
    """

    train_errorbars = None
    test_errorbars = None

    train_predictions = []
    test_predictions = []

    prediction_names = [
        f["name"] for f in get_model_prediction_features(model)
    ]

    train_predictions = model.predict(train_ds)

    if len(prediction_names) == 1:
        if (
            cb.conf.targets[
                cb.conf.target_names.index(prediction_names[0])
            ].type
            == "numerical"
        ):
            train_predictions = [train_predictions.flatten()]
    else:
        for i in range(len(train_predictions)):
            if (
                cb.conf.targets[
                    cb.conf.target_names.index(prediction_names[i])
                ].type
                == "numerical"
            ):
                train_predictions[i] = train_predictions[i].flatten()

    if test_ds:
        test_predictions = model.predict(test_ds)
        if len(prediction_names) == 1:
            if (
                cb.conf.targets[
                    cb.conf.target_names.index(prediction_names[0])
                ].type
                == "numerical"
            ):
                test_predictions = [test_predictions.flatten()]
        else:
            for i in range(len(test_predictions)):
                if (
                    cb.conf.targets[
                        cb.conf.target_names.index(prediction_names[i])
                    ].type
                    == "numerical"
                ):
                    test_predictions[i] = test_predictions[i].flatten()

    if plot:
        if test_ds is not None:
            cb.plots.plot_results_classification(
                train_labels,
                train_predictions,
                test_labels,
                test_predictions,
                model_name=model_name,
            )
            cb.plots.plot_results_regression(
                train_labels,
                train_predictions,
                test_labels,
                test_predictions,
                train_compositions=train_compositions,
                test_compositions=test_compositions,
                train_errorbars=train_errorbars,
                test_errorbars=test_errorbars,
                model_name=model_name,
            )
        else:
            cb.plots.plot_results_classification(
                train_labels, train_predictions, model_name=model_name
            )
            cb.plots.plot_results_regression(
                train_labels,
                train_predictions,
                train_compositions=train_compositions,
                test_compositions=test_compositions,
                train_errorbars=train_errorbars,
                model_name=model_name,
            )

    if test_ds is not None:
        return train_predictions, test_predictions
    else:
        return train_predictions


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

    data = cb.features.calculate_features(alloys, model=model)

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
