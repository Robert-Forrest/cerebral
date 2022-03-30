# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
from sklearn.model_selection import StratifiedKFold  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error
import metallurgy as mg

import cerebral as cb
from . import models
from . import plots
from . import metrics
from . import features
from . import loss


def kfolds_split(data, numFolds):
    data = data.copy()

    unique_composition_spaces = {}
    for _, row in data.iterrows():
        composition = mg.alloy.parse_composition(row['composition'])
        sorted_composition = sorted(list(composition.keys()))
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    foldSize = int(np.floor(len(shuffled_unique_compositions) / numFolds))

    folds = []
    for i in range(numFolds):
        trainingSetCompositions = []
        if i > 0:
            trainingSetCompositions = shuffled_unique_compositions[0:i*foldSize]

        testSetCompositions = shuffled_unique_compositions[i*foldSize:(
            i+1)*foldSize]

        if i < numFolds - 1:
            trainingSetCompositions.extend(
                shuffled_unique_compositions[(i+1)*foldSize:len(shuffled_unique_compositions)-1])

        trainingSet = []
        testSet = []
        for composition in trainingSetCompositions:
            trainingSet.extend(unique_composition_spaces[composition])
        for composition in testSetCompositions:
            testSet.extend(unique_composition_spaces[composition])

        folds.append([pd.DataFrame(trainingSet), pd.DataFrame(testSet)])

    return folds


def kfolds(originalData, save=False, plot=False):

    numFolds = cb.conf.kfolds.get("num_folds", 5)

    MADs = {}
    RMSDs = {}
    for feature in cb.conf.targets:
        if feature.type == 'numerical':
            MADs[feature.name] = metrics.meanAbsoluteDeviation(
                features.filter_masked(originalData[feature.name]))
            RMSDs[feature.name] = metrics.rootMeanSquareDeviation(
                features.filter_masked(originalData[feature.name]))

    MAEs = {}
    RMSEs = {}
    accuracies = {}
    f1s = {}
    for feature in cb.conf.targets:
        if feature.type == 'numerical':
            MAEs[feature.name] = []
            RMSEs[feature.name] = []
        else:
            accuracies[feature.name] = []
            f1s[feature.name] = []

    fold_test_labels = []
    fold_test_predictions = []

    folds = kfolds_split(originalData, numFolds)

    for foldIndex in range(numFolds):

        train_tmp = folds[foldIndex][0]
        test_tmp = folds[foldIndex][1]

        train_compositions = train_tmp.pop('composition')
        test_compositions = test_tmp.pop('composition')

        train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest = features.create_datasets(
            originalData, train=train_tmp, test=test_tmp)

        model = models.train_model(train_features, train_labels,
                                   sampleWeight,
                                   test_features=test_features, test_labels=test_labels, sampleWeight_test=sampleWeightTest,
                                   plot=plot, maxEpochs=cb.conf.train.get("max_epochs", 100), model_name=foldIndex)
        if save and not plot:
            models.save(
                model,
                cb.conf.output_directory +
                '/model' + str(foldIndex)
            )

        train_predictions, test_predictions = models.evaluate_model(
            model, train_ds, train_labels, test_ds, test_labels, plot=plot, model_name=foldIndex)

        fold_test_labels.append(test_labels)
        fold_test_predictions.append(test_predictions)

        for feature in cb.conf.targets:
            featureIndex = cb.conf.target_names.index(feature.name)
            if feature.type == 'numerical':

                test_labels_masked, test_predictions_masked = features.filter_masked(
                    test_labels[feature.name], test_predictions[featureIndex].flatten())

                MAEs[feature.name].append(metrics.calc_MAE(
                    test_labels_masked, test_predictions_masked))
                RMSEs[feature.name].append(metrics.calc_RMSE(
                    test_labels_masked, test_predictions_masked))
            else:
                test_labels_masked, test_predictions_masked = features.filter_masked(
                    test_labels[feature.name], test_predictions[featureIndex])

                accuracies[feature.name].append(metrics.calc_accuracy(
                    test_labels_masked, test_predictions_masked))
                f1s[feature.name].append(metrics.calc_f1(
                    test_labels_masked, test_predictions_masked))

    with open(cb.conf.output_directory + '/validation.dat', 'w') as validationFile:
        for feature in cb.conf.targets:
            if feature.type == 'numerical':
                validationFile.write('# ' + feature.name + '\n')
                validationFile.write('# MAD RMSD\n')
                validationFile.write(
                    str(MADs[feature.name]) + ' ' + str(RMSDs[feature.name]) + '\n')
                validationFile.write('# foldId MAE RMSE\n')
                for i in range(len(MAEs[feature.name])):
                    validationFile.write(
                        str(i) + ' ' + str(MAEs[feature.name][i]) + ' ' + str(RMSEs[feature.name][i]) + '\n')
                validationFile.write('# Mean \n')
                validationFile.write(
                    str(np.mean(MAEs[feature.name])) + ' ' + str(np.mean(RMSEs[feature.name])) + '\n')
                validationFile.write('# Standard Deviation \n')
                validationFile.write(
                    str(np.std(MAEs[feature.name])) + ' ' + str(np.std(RMSEs[feature.name])) + '\n\n')
            else:
                validationFile.write('# ' + feature.name + '\n')
                validationFile.write('# foldId accuracy f1\n')
                for i in range(len(accuracies[feature.name])):
                    validationFile.write(
                        str(i) + ' ' + str(accuracies[feature.name][i]) + ' ' + str(f1s[feature.name][i]) + '\n')
                validationFile.write('# Mean \n')
                validationFile.write(
                    str(np.mean(accuracies[feature.name])) + ' ' + str(np.mean(f1s[feature.name])) + '\n')
                validationFile.write('# Standard Deviation \n')
                validationFile.write(
                    str(np.std(accuracies[feature.name])) + ' ' + str(np.std(f1s[feature.name])) + '\n\n')

    plots.plot_results_regression_heatmap(
        fold_test_labels, fold_test_predictions)


def kfoldsEnsemble(originalData):

    kfolds(originalData, save=True, plot=True)

    compositions = originalData.pop('composition')

    train_ds, train_features, train_labels, sampleWeight = features.create_datasets(
        originalData)

    inputs = models.build_input_layers(train_features)
    outputs = []
    losses, metrics = models.setup_losses_and_metrics()

    numFolds = cb.conf.kfolds.get("num_folds", 5)

    submodel_outputs = []
    for k in range(numFolds):
        submodel = models.load(
            cb.conf.output_directory + '/model_' + str(k))
        submodel._name = "ensemble_" + str(k)
        for layer in submodel.layers:
            layer.trainable = False

        submodel_outputs.append(submodel(inputs))

    for i in range(len(cb.conf.targets)):

        submodel_output = [output[i] for output in submodel_outputs]

        submodels_merged = tf.keras.layers.concatenate(submodel_output)

        hidden = tf.keras.layers.Dense(64, activation="relu")(submodels_merged)

        output = None
        if cb.conf.targets[i].type == 'categorical':
            activation = "softmax"
            numNodes = 3
        else:
            activation = "softplus"
            numNodes = 1

        output = tf.keras.layers.Dense(
            numNodes,
            activation=activation,
            name=cb.conf.targets[i].name)(hidden)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(
        model, to_file=cb.conf.image_directory + 'model_ensemble.png', rankdir='LR')

    learning_rate = 0.01
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimiser,
        loss=losses,
        loss_weights={target['name']: target['weight']
                      for target in cb.conf.targets},
        metrics=metrics)

    model, history = models.fit(
        model, train_features, train_labels, sampleWeight, maxEpochs=cb.conf.train.get("max_epochs", 100))
    models.save(model, cb.conf.output_directory + '/model')

    plots.plot_training(history)
    train_predictions = models.evaluate_model(
        model, train_ds, train_labels, train_compositions=compositions)
