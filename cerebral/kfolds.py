"""Module providing k-folds cross-validation functionality."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
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
        sorted_composition = sorted(list(composition.keys()))
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


def kfolds(originalData: pd.DataFrame, save: bool = False, plot: bool = False):
    """Performs k-folds cross-validation to evaluate a model.

    :group: kfolds

    Parameters
    ----------

    originalData
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
                cb.features.filter_masked(originalData[feature.name])
            )
            RMSDs[feature.name] = cb.metrics.rootMeanSquareDeviation(
                cb.features.filter_masked(originalData[feature.name])
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

    fold_test_labels = []
    fold_test_predictions = []

    folds = kfolds_split(originalData, num_folds)

    for foldIndex in range(num_folds):

        train_tmp = folds[foldIndex][0]
        test_tmp = folds[foldIndex][1]

        train_tmp.pop("composition")
        test_tmp.pop("composition")

        (
            train_ds,
            test_ds,
            train_features,
            test_features,
            train_labels,
            test_labels,
            sampleWeight,
            sampleWeightTest,
        ) = cb.features.create_datasets(
            originalData, cb.conf.targets, train=train_tmp, test=test_tmp
        )

        cb.conf.model_name = foldIndex
        model = cb.models.train_model(
            train_features,
            train_labels,
            sampleWeight,
            test_features=test_features,
            test_labels=test_labels,
            sampleWeight_test=sampleWeightTest,
            plot=plot,
            maxEpochs=cb.conf.train.get("max_epochs", 100),
        )
        if save and not plot:
            cb.models.save(
                model, cb.conf.output_directory + "/model" + str(foldIndex)
            )

        train_predictions, test_predictions = cb.models.evaluate_model(
            model,
            train_ds,
            train_labels,
            test_ds,
            test_labels,
            plot=plot,
        )

        fold_test_labels.append(test_labels)
        fold_test_predictions.append(test_predictions)

        for feature in cb.conf.targets:
            featureIndex = cb.conf.target_names.index(feature.name)
            if feature.type == "numerical":

                (
                    test_labels_masked,
                    test_predictions_masked,
                ) = cb.features.filter_masked(
                    test_labels[feature.name],
                    test_predictions[featureIndex].flatten(),
                )

                MAEs[feature.name].append(
                    cb.metrics.calc_MAE(
                        test_labels_masked, test_predictions_masked
                    )
                )
                RMSEs[feature.name].append(
                    cb.metrics.calc_RMSE(
                        test_labels_masked, test_predictions_masked
                    )
                )
            else:
                (
                    test_labels_masked,
                    test_predictions_masked,
                ) = cb.features.filter_masked(
                    test_labels[feature.name], test_predictions[featureIndex]
                )

                accuracies[feature.name].append(
                    cb.metrics.calc_accuracy(
                        test_labels_masked, test_predictions_masked
                    )
                )
                f1s[feature.name].append(
                    cb.metrics.calc_f1(
                        test_labels_masked, test_predictions_masked
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
        fold_test_labels, fold_test_predictions
    )


def kfoldsEnsemble(originalData: pd.DataFrame):
    """Construct an ensemble model using multiple k-folds submodels.
    See Section 4.2 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: kfolds

    Parameters
    ----------

    originalData
        The dataset used to train models, which will be folded into multiple
        training and testing subsets.

    """

    kfolds(originalData, save=True, plot=True)

    compositions = originalData.pop("composition")

    (
        train_ds,
        train_features,
        train_labels,
        sampleWeight,
    ) = cb.features.create_datasets(originalData, cb.conf.targets)

    inputs = cb.models.build_input_layers(train_features)
    outputs = []
    losses, metrics = cb.models.setup_losses_and_metrics()

    num_folds = cb.conf.kfolds.get("num_folds", 5)

    submodel_outputs = []
    for k in range(num_folds):
        submodel = cb.models.load(
            cb.conf.output_directory + "/model_" + str(k)
        )
        submodel._name = "ensemble_" + str(k)
        for layer in submodel.layers:
            layer.trainable = False

        submodel_outputs.append(submodel(inputs))

    for i in range(len(cb.conf.targets)):

        submodel_output = [output[i] for output in submodel_outputs]

        submodels_merged = tf.keras.layers.concatenate(submodel_output)

        hidden = tf.keras.layers.Dense(64, activation="relu")(submodels_merged)

        output = None
        if cb.conf.targets[i].type == "categorical":
            activation = "softmax"
            numNodes = 3
        else:
            activation = "softplus"
            numNodes = 1

        output = tf.keras.layers.Dense(
            numNodes, activation=activation, name=cb.conf.targets[i].name
        )(hidden)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(
        model,
        to_file=cb.conf.image_directory + "model_ensemble.png",
        rankdir="LR",
    )

    learning_rate = 0.01
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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
        train_features,
        train_labels,
        sampleWeight,
        maxEpochs=cb.conf.train.get("max_epochs", 100),
    )
    cb.models.save(model, cb.conf.output_directory + "/model")

    cb.plots.plot_training(history)
    train_predictions = cb.models.evaluate_model(
        model, train_ds, train_labels, train_compositions=compositions
    )
