# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np  # pylint: disable=import-error

import cerebral as cb
from . import plots
from . import features
from . import models
from . import metrics


def permutation():

    numPermutations = cb.conf.get("permutations", 5)

    model_name = "kfoldsEnsemble"

    open(cb.conf.output_directory + '/permutedFeatures.dat', 'w')

    model = models.load(cb.conf.output_directory+'/model')
    originalData = features.load_data(model=model)

    permutationImportance = {}
    for permutedFeature in ['none'] + list(originalData.columns):
        if permutedFeature not in features.predictableFeatures and permutedFeature != 'composition':

            permutationImportance[permutedFeature] = {}
            for feature in features.predictableFeatures:
                permutationImportance[permutedFeature][feature] = []

            for k in range(numPermutations):

                data = originalData.copy()
                if permutedFeature != 'none':
                    data[permutedFeature] = np.random.permutation(
                        data[permutedFeature].values)

                train_ds, train_features, labels, sampleWeight = features.create_datasets(
                    data)

                predictions = models.evaluate_model(
                    model, train_ds, labels, plot=False)

                for feature in features.predictableFeatures:
                    featureIndex = features.predictableFeatures.index(feature)
                    if feature != 'GFA':

                        labels_masked, predictions_masked = features.filter_masked(
                            labels[feature], predictions[featureIndex].flatten())

                        permutationImportance[permutedFeature][feature].append(metrics.calc_MAE(
                            labels_masked, predictions_masked))

                    else:

                        labels_masked, predictions_masked = features.filter_masked(
                            labels[feature], predictions[featureIndex])

                        permutationImportance[permutedFeature][feature].append(metrics.calc_accuracy(
                            labels_masked, predictions_masked))
                if permutedFeature == 'none':
                    for feature in features.predictableFeatures:
                        permutationImportance[permutedFeature][feature] = permutationImportance[permutedFeature][feature][0]
                    break

            if permutedFeature != 'none':
                for feature in features.predictableFeatures:
                    averageScore = 0
                    for i in range(numPermutations):
                        averageScore += permutationImportance[permutedFeature][feature][i]
                    averageScore /= numPermutations
                    if feature != 'GFA':
                        permutationImportance[permutedFeature][feature] = max(0, averageScore -
                                                                              permutationImportance['none'][feature])
                    else:
                        permutationImportance[permutedFeature][feature] = max(
                            0, permutationImportance['none'][feature] - averageScore)

            with open(cb.conf.output_directory + '/permutedFeatures.dat', 'a') as resultsFile:
                for feature in features.predictableFeatures:
                    resultsFile.write(permutedFeature + ' ' + feature + ' ' +
                                      " " + str(permutationImportance[permutedFeature][feature]) + '\n')

    del permutationImportance['none']
    plots.plot_feature_permutation(permutationImportance)
