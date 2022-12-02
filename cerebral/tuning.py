"""Module providing hyperparameter tuning functionality."""

import keras_tuner as kt
import tensorflow as tf

import cerebral as cb


class HyperModel(kt.HyperModel):

    """Variant of the keras-tuner HyperModel class

    :group: tuning
    """

    def __init__(self, train_ds):
        self.train_ds = train_ds

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
            self.train_ds,
            num_shared_layers=num_shared_layers,
            num_specific_layers=num_specific_layers,
            units_per_layer=units_per_layer,
        )
        return model


def tune(
    data,
    tuner="hyperband",
):
    """Perform hyperparameter tuning using kerastuner.

    :group: tuning
    """

    train_ds = cb.features.create_datasets(data, targets=cb.conf.targets)

    if tuner == "hyperband":
        tuner = kt.tuners.Hyperband(
            HyperModel(train_ds),
            objective="loss",
            factor=3,
            max_epochs=cb.conf.train.get("max_epochs", 1000),
            hyperband_iterations=5,
            directory=cb.conf.output_directory + "/tuning",
            project_name="GFA",
        )
    elif tuner == "bayesian":
        tuner = kt.tuners.BayesianOptimization(
            HyperModel(train_ds),
            objective="loss",
            max_trials=1000,
            directory=cb.conf.output_directory + "/tuning",
            project_name="GFA",
        )

    patience = 100
    min_delta = 0.01

    tuner.search(
        train_ds,
        batch_size=cb.conf.train.get("batch_size", 256),
        epochs=cb.conf.train.get("max_epochs", 1000),
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
    )
