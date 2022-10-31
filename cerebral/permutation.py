"""Module providing feature permutation functionality."""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)
import numpy as np

import cerebral as cb


def permutation(postprocess=None):
    """Performs feature permutation analysis to identify important features. See
    Section 5 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: permutation

    Parameters
    ----------

    postprocess
        A function to run on the data after loading.

    """

    num_permutations = cb.conf.get("permutations", 5)

    open(cb.conf.output_directory + "/permuted_features.dat", "w")

    model = cb.models.load(cb.conf.output_directory + "/model")
    originalData = cb.features.load_data(
        model=model, plot=False, postprocess=postprocess
    )

    permutation_importance = {}
    for permuted_feature in ["none"] + list(originalData.columns):
        if (
            permuted_feature not in cb.conf.target_names
            and permuted_feature != "composition"
        ):

            permutation_importance[permuted_feature] = {}
            for feature in cb.conf.targets:
                permutation_importance[permuted_feature][feature.name] = []

            for k in range(num_permutations):

                data = originalData.copy()
                if permuted_feature != "none":
                    data[permuted_feature] = np.random.permutation(
                        data[permuted_feature].values
                    )

                (
                    train_ds,
                    train_features,
                    labels,
                    sampleWeight,
                ) = cb.features.create_datasets(data)

                predictions = cb.models.evaluate_model(
                    model, train_ds, labels, plot=False
                )

                for feature in cb.conf.targets:
                    feature_index = cb.conf.target_names.index(feature.name)
                    if feature.type == "numerical":

                        (
                            labels_masked,
                            predictions_masked,
                        ) = cb.features.filter_masked(
                            labels[feature.name],
                            predictions[feature_index].flatten(),
                        )

                        permutation_importance[permuted_feature][
                            feature.name
                        ].append(
                            cb.metrics.calc_MAE(
                                labels_masked, predictions_masked
                            )
                        )

                    else:

                        (
                            labels_masked,
                            predictions_masked,
                        ) = cb.features.filter_masked(
                            labels[feature.name], predictions[feature_index]
                        )

                        permutation_importance[permuted_feature][
                            feature.name
                        ].append(
                            cb.metrics.calc_accuracy(
                                labels_masked, predictions_masked
                            )
                        )

                if permuted_feature == "none":
                    for feature in cb.conf.targets:
                        permutation_importance[permuted_feature][
                            feature.name
                        ] = permutation_importance[permuted_feature][
                            feature.name
                        ][
                            0
                        ]
                    break

            if permuted_feature != "none":
                for feature in cb.conf.targets:
                    average_score = 0
                    for i in range(num_permutations):
                        average_score += permutation_importance[
                            permuted_feature
                        ][feature.name][i]
                    average_score /= num_permutations
                    if feature.type == "numerical":
                        permutation_importance[permuted_feature][
                            feature.name
                        ] = max(
                            0,
                            average_score
                            - permutation_importance["none"][feature.name],
                        )
                    else:
                        permutation_importance[permuted_feature][
                            feature.name
                        ] = max(
                            0,
                            permutation_importance["none"][feature.name]
                            - average_score,
                        )

            with open(
                cb.conf.output_directory + "/permuted_features.dat", "a"
            ) as resultsFile:
                for feature in cb.conf.targets:
                    resultsFile.write(
                        permuted_feature
                        + " "
                        + feature.name
                        + " "
                        + " "
                        + str(
                            permutation_importance[permuted_feature][
                                feature.name
                            ]
                        )
                        + "\n"
                    )

    del permutation_importance["none"]
    cb.plots.plot_feature_permutation(permutation_importance)
