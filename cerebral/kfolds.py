"""Module providing k-folds cross-validation functionality."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import os
from typing import List

import tensorflow as tf
import pandas as pd
import numpy as np
import metallurgy as mg

import cerebral as cb


def kfolds_split(
    data: pd.DataFrame, num_folds: int
) -> List[List[pd.DataFrame]]:
    """Split a dataframe into several folds of training and test subsets.

    :group: kfolds

    data
       DataFrame to be split.
    num_folds
       Number of folds to split into.

    """

    data = data.copy()

    unique_composition_spaces = {}
    for _, row in data.iterrows():
        composition = mg.alloy.parse_composition(row["composition"])
        elements = []
        for element in composition:
            if composition[element] > 0.02:
                elements.append(element)

        sorted_composition = sorted(elements)
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    foldSize = int(np.floor(len(shuffled_unique_compositions) / num_folds))

    folds = []
    for i in range(num_folds):
        trainingSetCompositions = []
        if i > 0:
            trainingSetCompositions = shuffled_unique_compositions[
                0 : i * foldSize
            ]

        testSetCompositions = shuffled_unique_compositions[
            i * foldSize : (i + 1) * foldSize
        ]

        if i < num_folds - 1:
            trainingSetCompositions.extend(
                shuffled_unique_compositions[
                    (i + 1) * foldSize : len(shuffled_unique_compositions) - 1
                ]
            )

        trainingSet = []
        testSet = []
        for composition in trainingSetCompositions:
            trainingSet.extend(unique_composition_spaces[composition])
        for composition in testSetCompositions:
            testSet.extend(unique_composition_spaces[composition])

        folds.append([pd.DataFrame(trainingSet), pd.DataFrame(testSet)])

    return folds


def kfolds(data: pd.DataFrame, save: bool = False, plot: bool = False):
    """Performs k-folds cross-validation to evaluate a model.

    :group: kfolds

    Parameters
    ----------

    data
        The dataset used to train models, which will be folded into multiple
        training and testing subsets.
    save
        If True, each submodel will be saved to disk.
    plot
        If True, training metrics for each submodel will be plotted.

    """

    num_folds = cb.conf.kfolds.get("num_folds", 5)

    MADs = {}
    RMSDs = {}
    for feature in cb.conf.targets:
        if feature.type == "numerical":
            MADs[feature.name] = cb.metrics.meanAbsoluteDeviation(
                cb.features.filter_masked(data[feature.name])
            )
            RMSDs[feature.name] = cb.metrics.rootMeanSquareDeviation(
                cb.features.filter_masked(data[feature.name])
            )

    MAEs = {}
    RMSEs = {}
    accuracies = {}
    f1s = {}
    for feature in cb.conf.targets:
        if feature.type == "numerical":
            MAEs[feature.name] = []
            RMSEs[feature.name] = []
        else:
            accuracies[feature.name] = []
            f1s[feature.name] = []

    fold_test_truth = []
    fold_test_predictions = []

    folds = kfolds_split(data, num_folds)

    for foldIndex in range(num_folds):

        train_tmp = folds[foldIndex][0]
        test_tmp = folds[foldIndex][1]

        train_ds, test_ds = cb.features.create_datasets(
            data, cb.conf.targets, train=train_tmp, test=test_tmp
        )

        cb.conf.model_name = "fold_" + str(foldIndex)

        model, history = cb.models.compile_and_fit(
            train_ds,
            test_ds=test_ds,
        )

        (
            train_evaluation,
            test_evaluation,
            metrics,
        ) = cb.models.evaluate_model(model, train_ds, test_ds=test_ds)

        fold_test_truth.append(test_evaluation["truth"])
        fold_test_predictions.append(test_evaluation["predictions"])

        for feature in cb.conf.targets:
            if feature.type == "numerical":

                (
                    test_truth_masked,
                    test_predictions_masked,
                ) = cb.features.filter_masked(
                    test_evaluation["truth"][feature.name],
                    test_evaluation["predictions"][feature.name],
                )

                MAEs[feature.name].append(
                    cb.metrics.calc_MAE(
                        test_truth_masked, test_predictions_masked
                    )
                )
                RMSEs[feature.name].append(
                    cb.metrics.calc_RMSE(
                        test_truth_masked, test_predictions_masked
                    )
                )
            else:
                (
                    test_truth_masked,
                    test_predictions_masked,
                ) = cb.features.filter_masked(
                    test_evaluation["truth"][feature.name],
                    test_evaluation["predictions"][feature.name],
                )

                accuracies[feature.name].append(
                    cb.metrics.calc_accuracy(
                        test_truth_masked, test_predictions_masked
                    )
                )
                f1s[feature.name].append(
                    cb.metrics.calc_f1(
                        test_truth_masked, test_predictions_masked
                    )
                )

    with open(
        cb.conf.output_directory + "/validation.dat", "w"
    ) as validationFile:
        for feature in cb.conf.targets:
            if feature.type == "numerical":
                validationFile.write("# " + feature.name + "\n")
                validationFile.write("# MAD RMSD\n")
                validationFile.write(
                    str(MADs[feature.name])
                    + " "
                    + str(RMSDs[feature.name])
                    + "\n"
                )
                validationFile.write("# foldId MAE RMSE\n")
                for i in range(len(MAEs[feature.name])):
                    validationFile.write(
                        str(i)
                        + " "
                        + str(MAEs[feature.name][i])
                        + " "
                        + str(RMSEs[feature.name][i])
                        + "\n"
                    )
                validationFile.write("# Mean \n")
                validationFile.write(
                    str(np.mean(MAEs[feature.name]))
                    + " "
                    + str(np.mean(RMSEs[feature.name]))
                    + "\n"
                )
                validationFile.write("# Standard Deviation \n")
                validationFile.write(
                    str(np.std(MAEs[feature.name]))
                    + " "
                    + str(np.std(RMSEs[feature.name]))
                    + "\n\n"
                )
            else:
                validationFile.write("# " + feature.name + "\n")
                validationFile.write("# foldId accuracy f1\n")
                for i in range(len(accuracies[feature.name])):
                    validationFile.write(
                        str(i)
                        + " "
                        + str(accuracies[feature.name][i])
                        + " "
                        + str(f1s[feature.name][i])
                        + "\n"
                    )
                validationFile.write("# Mean \n")
                validationFile.write(
                    str(np.mean(accuracies[feature.name]))
                    + " "
                    + str(np.mean(f1s[feature.name]))
                    + "\n"
                )
                validationFile.write("# Standard Deviation \n")
                validationFile.write(
                    str(np.std(accuracies[feature.name]))
                    + " "
                    + str(np.std(f1s[feature.name]))
                    + "\n\n"
                )

    cb.plots.plot_results_regression_heatmap(
        fold_test_truth, fold_test_predictions
    )


def kfoldsEnsemble(data: pd.DataFrame):
    """Construct an ensemble model using multiple k-folds submodels.
    See Section 4.2 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: kfolds

    Parameters
    ----------

    data
        The dataset used to train models, which will be folded into multiple
        training and testing subsets.

    """

    kfolds(data, save=True, plot=True)

    train_ds = cb.features.create_datasets(data, cb.conf.targets)

    inputs = cb.models.build_input_layers(train_ds)
    outputs = []
    losses, metrics = cb.models.setup_losses_and_metrics()

    num_folds = cb.conf.kfolds.get("num_folds", 5)

    submodel_outputs = []
    for k in range(num_folds):
        submodel = cb.models.load(cb.conf.output_directory + "/fold_" + str(k))
        submodel._name = "fold_" + str(k)
        for layer in submodel.layers:
            layer.trainable = False

        submodel_outputs.append(submodel(inputs))

    for i in range(len(cb.conf.targets)):

        if len(cb.conf.targets) > 1:
            submodel_output = [output[i] for output in submodel_outputs]
            submodels_merged = tf.keras.layers.Concatenate()(submodel_output)
        else:
            submodels_merged = tf.keras.layers.Concatenate()(submodel_outputs)

        hidden = tf.keras.layers.Dense(
            64, activation=cb.conf.train.get("activation", "relu")
        )(submodels_merged)

        output = None
        if cb.conf.targets[i]["type"] == "categorical":
            activation = "softmax"
            num_nodes = len(cb.conf.targets[i]["classes"])
        else:
            activation = "softplus"
            num_nodes = 1

        output = tf.keras.layers.Dense(
            num_nodes, activation=activation, name=cb.conf.targets[i].name
        )(hidden)
        outputs.append(output)

    cb.conf.model_name = "ensemble"
    os.makedirs(cb.conf.output_directory + "ensemble", exist_ok=True)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(
        model,
        to_file=cb.conf.output_directory + "ensemble/ensemble.png",
        rankdir="LR",
    )

    optimiser = tf.keras.optimizers.Adam(
        learning_rate=cb.conf.train.get("learning_rate", 0.01)
    )

    model.compile(
        optimizer=optimiser,
        loss=losses,
        loss_weights={
            target["name"]: target["weight"] for target in cb.conf.targets
        },
        metrics=metrics,
    )

    model, history = cb.models.fit(
        model,
        train_ds,
        max_epochs=2000,
    )
    cb.models.save(model, cb.conf.output_directory + "/ensemble")

    cb.plots.plot_training(history)

    train_evaluation, metrics = cb.models.evaluate_model(model, train_ds)
