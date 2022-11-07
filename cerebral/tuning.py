"""Module providing hyperparameter tuning functionality."""

import keras_tuner as kt
import tensorflow as tf

import cerebral as cb


class HyperModel(kt.HyperModel):

    """Variant of the keras-tuner HyperModel class

    :group: tuning
    """

    def __init__(self, train_features):
        self.train_features = train_features

    def build(self, hp):

        num_shared_layers = hp.Int(
            "num_shared_layers", min_value=2, max_value=10, step=1
        )
        num_specific_layers = hp.Int(
            "num_specific_layers", min_value=2, max_value=10, step=1
        )

        units_per_layer = hp.Choice(
            "units_per_layer", values=[8, 16, 32, 64, 128, 256, 512]
        )

        model = cb.models.build_model(
            train_features=self.train_features,
            num_shared_layers=num_shared_layers,
            num_specific_layers=num_specific_layers,
            units_per_layer=units_per_layer,
        )
        return model


def tune(
    train_features,
    train_labels,
    sampleWeight,
    test_features=None,
    test_labels=None,
    sampleWeightTest=None,
    tuner="bayesian",
):
    """Perform hyperparameter tuning using kerastuner.

    :group: tuning
    """

    if tuner == "bayesian":
        tuner = kt.tuners.BayesianOptimization(
            HyperModel(train_features),
            objective="loss",
            max_trials=1000,
            directory=cb.conf.output_directory + "/models",
            project_name="GFA",
        )

    elif tuner == "hyperband":
        tuner = kt.tuners.Hyperband(
            HyperModel(train_features),
            objective="loss",
            factor=3,
            max_epochs=1000,
            hyperband_iterations=1,
            directory=cb.conf.output_directory + "/models",
            project_name="GFA",
        )

    patience = 100
    min_delta = 0.01

    xTrain = {}
    for feature in train_features:
        xTrain[feature] = train_features[feature]

    yTrain = {}
    for feature in cb.cb.conf.targets:
        yTrain[feature] = train_labels[feature]

    if test_features is not None:
        xTest = {}
        for feature in test_features:
            xTest[feature] = test_features[feature]

        yTest = {}
        for feature in cb.cb.conf.targets:
            yTest[feature] = test_labels[feature]

        tuner.search(
            x=xTrain,
            y=yTrain,
            batch_size=cb.features.batch_size,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=patience,
                    min_delta=min_delta,
                    mode="auto",
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.75,
                    patience=patience // 3,
                    mode="auto",
                    min_delta=min_delta * 10,
                    cooldown=patience // 4,
                    min_lr=0,
                ),
            ],
            verbose=2,
            validation_data=(xTest, yTest, sampleWeightTest),
            sample_weight=sampleWeight,
        )

    else:

        tuner.search(
            x=xTrain,
            y=yTrain,
            batch_size=cb.features.batch_size,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=patience,
                    min_delta=min_delta,
                    mode="auto",
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.75,
                    patience=patience // 3,
                    mode="auto",
                    min_delta=min_delta * 10,
                    cooldown=patience // 4,
                    min_lr=0,
                ),
            ],
            verbose=2,
            sample_weight=sampleWeight,
        )
