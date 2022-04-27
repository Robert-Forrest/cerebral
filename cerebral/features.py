import re
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np  # pylint: disable=import-error
import scipy.stats as stats
from scipy.cluster import hierarchy
import pandas as pd  # pylint: disable=import-error
import tensorflow as tf  # pylint: disable=import-error
from sklearn.feature_selection import VarianceThreshold  # pylint: disable=import-error
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from decimal import Decimal
from collections.abc import Iterable

import elementy
import cerebral as cb
import metallurgy as mg

from . import models
from . import plots


maskValue = -1

units = {
    'Dmax': 'mm',
    'Tl': 'K',
    'Tg': 'K',
    'Tx': 'K',
    'deltaT': 'K',
    'price_linearmix': "\\$/kg",
    'price': "\\$/kg",
    'mixing_enthalpy': 'kJ/mol',
    'mixing_Gibbs_free_energy': 'kJ/mol'
}
inverse_units = {}
for feature in units:
    if "/" not in units[feature]:
        inverse_units[feature] = "1/" + units[feature]
    else:
        split_units = units[feature].split('/')
        inverse_units[feature] = split_units[1] + "/" + split_units[0]


def calculate_compositions(data):
    compositions = []
    columns_to_drop = []
    for _, row in data.iterrows():
        composition = {}
        for column in data.columns:
            if column not in cb.conf.target_names:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)
                if row[column] > 0:
                    composition[column] = row[column] / 100.0
        composition = mg.Alloy(composition, rescale=False)
        compositions.append(composition.to_string())

    data['composition'] = compositions
    for column in columns_to_drop:
        data = data.drop(column, axis='columns')
    return data


def camelCaseToSentence(string):
    if(string == string.upper()):
        return string
    else:
        tmp = re.sub(r'([A-Z][a-z])', r" \1",
                     re.sub(r'([A-Z]+)', r"\1", string))
        return tmp[0].upper() + tmp[1:]


def prettyName(feature):
    if cb.conf is not None:
        if feature in cb.conf.pretty_feature_names:
            return r'$'+cb.conf.pretty_features[cb.conf.pretty_feature_names.index(feature)].pretty+'$'

    name = ""
    featureParts = feature.split('_')
    if 'linearmix' in feature or 'deviation' in feature:
        if len(featureParts) > 1:
            if featureParts[-1] == 'linearmix':
                name = r'$\Sigma$ '
            elif featureParts[-1] == 'deviation':
                name = r'$\delta$ '
        name += ' '.join(word.title() for word in featureParts[0:-1])
    else:
        name += ' '.join(word.title() for word in featureParts)
    return name


def calculate_features(
        data,
        dropCorrelatedFeatures=True, plot=False,
        additionalFeatures=[], requiredFeatures=[],
        merge_duplicates=True, model=None):

    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, Iterable) and not isinstance(data, (str, dict)):
            data = [data]

        parsed_data = []
        for i in range(len(data)):
            alloy = data[i]
            if not isinstance(data[i], mg.Alloy):
                alloy = mg.Alloy(data[i], rescale=False)
            parsed_data.append(alloy.to_string())

        data = pd.DataFrame(parsed_data, columns=['composition'])

    if model is not None:
        dropCorrelatedFeatures = False
        merge_duplicates = False

        targets = models.get_model_prediction_features(model)
        target_names = [target['name'] for target in targets]

        input_features = models.get_model_input_features(model)
        basic_features = []
        complex_features = []

        for feature in input_features:
            actual_feature = feature.split(
                "_linearmix")[0].split('_deviation')[0]

            if '_linearmix' in feature or '_deviation' in feature:
                if actual_feature not in basic_features:
                    basic_features.append(actual_feature)
            else:
                if feature not in complex_features:
                    complex_features.append(feature)

    else:
        basic_features = cb.conf.basic_features
        complex_features = cb.conf.complex_features
        targets = cb.conf.targets
        target_names = cb.conf.target_names

    for additionalFeature in additionalFeatures:
        actual_feature = additionalFeature.split(
            "_linearmix")[0].split('_deviation')[0]
        if (actual_feature not in basic_features
            and actual_feature not in complex_features
                and actual_feature not in target_names):
            basic_features.append(actual_feature)

    if len(requiredFeatures) > 0:
        dropCorrelatedFeatures = False

        for feature in requiredFeatures:

            if "_linearmix" in feature:
                actual_feature = feature.split("_linearmix")[0]
                if actual_feature not in basic_features and actual_feature not in complex_features and feature not in complex_features:
                    basic_features.append(actual_feature)

            elif "_deviation" in feature:
                actual_feature = feature.split("_deviation")[0]
                if actual_feature not in basic_features and actual_feature not in complex_features and feature not in complex_features:
                    basic_features.append(actual_feature)

            else:
                if feature not in complex_features:
                    complex_features.append(feature)

    feature_values = {}
    complex_feature_values = {}

    for feature in basic_features:
        feature_values[feature] = {
            'linearmix': [],
            'deviation': []
        }
        units[feature + '_deviation'] = "%"

    for feature in complex_features:
        complex_feature_values[feature] = []

    for feature in targets:
        if(feature['type'] == 'categorical' and feature['name'] in data.columns):
            data[feature['name']] = data[feature['name']].map(
                {feature.classes[i]: i for i in range(len(feature.classes))}
            )
            data[feature['name']] = data[feature['name']].fillna(maskValue)
            data[feature['name']] = data[feature['name']].astype(np.int64)

    for i, row in data.iterrows():

        composition = mg.alloy.parse_composition(row['composition'])

        for feature in basic_features:

            if 'linearmix' in feature_values[feature]:
                feature_values[feature]['linearmix'].append(
                    mg.linear_mixture(composition, feature))

            if 'deviation' in feature_values[feature]:
                feature_values[feature]['deviation'].append(
                    mg.deviation(composition, feature))

        for feature in complex_feature_values:
            complex_feature_values[feature].append(
                mg.calculate(composition, feature))

    for feature in feature_values:
        for kind in feature_values[feature]:
            if len(feature_values[feature][kind]) == len(data.index):
                data[feature + '_' + kind] = feature_values[feature][kind]
    for feature in complex_features:
        if len(complex_feature_values[feature]) == len(data.index):
            data[feature] = complex_feature_values[feature]

    data = data.fillna(maskValue)

    if merge_duplicates:
        data = data.drop_duplicates()
        to_drop = []
        seen_compositions = []
        duplicate_compositions = {}
        for i, row in data.iterrows():
            alloy = mg.Alloy(row['composition'], rescale=False)
            composition = alloy.to_string()

            if(abs(1-sum(alloy.composition.values())) > 0.01):
                print("Invalid composition:", row['composition'], i)
                to_drop.append(i)

            elif(composition in seen_compositions):
                if composition not in duplicate_compositions:
                    duplicate_compositions[composition] = [
                        data.iloc[seen_compositions.index(composition)]
                    ]
                duplicate_compositions[composition].append(row)
                to_drop.append(i)
            seen_compositions.append(composition)

        data = data.drop(to_drop)

        to_drop = []
        for i, row in data.iterrows():
            composition = mg.Alloy(
                row['composition'], rescale=False).to_string()

            if composition in duplicate_compositions:
                to_drop.append(i)

        data = data.drop(to_drop)

        deduplicated_rows = []
        for composition in duplicate_compositions:

            averaged_features = {}
            num_contributions = {}
            for feature in duplicate_compositions[composition][0].keys():
                if feature != 'composition':
                    averaged_features[feature] = 0
                    num_contributions[feature] = 0

            for i in range(len(duplicate_compositions[composition])):
                for feature in averaged_features:
                    if duplicate_compositions[composition][i][feature] != maskValue and not pd.isnull(
                            duplicate_compositions[composition][i][feature]):

                        averaged_features[feature] += duplicate_compositions[composition][i][feature]
                        num_contributions[feature] += 1

            for feature in averaged_features:
                if num_contributions[feature] == 0:
                    averaged_features[feature] = maskValue
                elif num_contributions[feature] > 1:
                    averaged_features[feature] /= num_contributions[feature]

            averaged_features['composition'] = composition

            deduplicated_rows.append(
                pd.DataFrame(averaged_features, index=[0]))

        if(len(deduplicated_rows) > 0):
            deduplicated_data = pd.concat(deduplicated_rows, ignore_index=True)
            data = pd.concat([data, deduplicated_data], ignore_index=True)

    if plot:
        plots.plot_correlation(data)
        plots.plot_feature_variation(data)

    droppedFeatures = []
    if dropCorrelatedFeatures:

        staticFeatures = []
        varianceCheckData = data.drop('composition', axis='columns')
        for feature in data.columns:
            if feature in [t['name'] for t in targets]:
                varianceCheckData = varianceCheckData.drop(
                    feature, axis='columns')

        quartileDiffusions = {}
        for feature in varianceCheckData.columns:

            Q1 = np.percentile(varianceCheckData[feature], 25)
            Q3 = np.percentile(varianceCheckData[feature], 75)

            coefficient = 0
            if np.abs(Q1 + Q3) > 0:
                coefficient = np.abs((Q3 - Q1) / (Q3 + Q1))
            quartileDiffusions[feature] = coefficient

            if coefficient < 0.1:
                staticFeatures.append(feature)

        print("Dropping static features:", staticFeatures)
        for feature in staticFeatures:
            varianceCheckData = varianceCheckData.drop(
                feature, axis='columns')

        correlation = np.array(varianceCheckData.corr())

        correlatedDroppedFeatures = []
        for i in range(len(correlation) - 1):
            if varianceCheckData.columns[i] not in correlatedDroppedFeatures:
                for j in range(i + 1, len(correlation)):
                    if varianceCheckData.columns[j] not in correlatedDroppedFeatures:
                        if np.abs(correlation[i][j]) >= cb.conf.train.get("correlation_threshold", 0.8):

                            if sum(np.abs(correlation[i])) < sum(
                                    np.abs(correlation[j])):
                                print(varianceCheckData.columns[j],
                                      sum(np.abs(correlation[j])), "beats",
                                      varianceCheckData.columns[i],
                                      sum(np.abs(correlation[i])))
                                correlatedDroppedFeatures.append(
                                    varianceCheckData.columns[i])
                                break
                            else:
                                print(varianceCheckData.columns[i], sum(np.abs(correlation[i])),
                                      "beats", varianceCheckData.columns[j], sum(np.abs(correlation[j])))
                                correlatedDroppedFeatures.append(
                                    varianceCheckData.columns[j])

        droppedFeatures = staticFeatures + correlatedDroppedFeatures

    if len(droppedFeatures) > 0:
        for feature in droppedFeatures:
            if feature in data.columns:
                data = data.drop(feature, axis='columns')

        if plot:
            plots.plot_correlation(data, suffix="filtered")
            plots.plot_feature_variation(data, suffix="filtered")

    if len(requiredFeatures) > 0:

        for feature in data.columns:
            trueFeatureName = feature.split(
                '_linearmix')[0].split('_deviation')[0]

            if (feature not in requiredFeatures
                and feature != 'composition'
                and feature not in target_names
                and feature not in additionalFeatures
                    and trueFeatureName not in additionalFeatures):

                print("Dropping", feature)
                data = data.drop(feature, axis='columns')

    return data.copy()


def train_test_split(data, trainPercentage=0.75):
    data = data.copy()

    unique_composition_spaces = {}
    for _, row in data.iterrows():
        composition = mg.alloy.parse_composition(row['composition'])
        sorted_composition = sorted(list(composition.keys()))
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    numTraining = np.ceil(
        int(trainPercentage * len(unique_composition_spaces)))

    trainingSet = []
    testSet = []

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    for i in range(len(shuffled_unique_compositions)):
        compositions = unique_composition_spaces[shuffled_unique_compositions[i]]
        if i < numTraining:
            trainingSet.extend(compositions)
        else:
            testSet.extend(compositions)

    return pd.DataFrame(trainingSet), pd.DataFrame(testSet)


def df_to_dataset(dataframe, targets=[]):
    dataframe = dataframe.copy()

    labelNames = []
    for feature in targets:
        if feature['name'] in dataframe.columns:
            labelNames.append(feature['name'])

    if len(labelNames) > 0:
        labels = pd.concat([dataframe.pop(x)
                            for x in labelNames], axis=1)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    batch_size = 1024
    if cb.conf:
        if cb.conf.get("train", None) is not None:
            batch_size = cb.conf.train.get('batch_size', batch_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    ds = ds.cache()

    return ds


def generate_sample_weights(labels, classFeature, classWeights):
    sampleWeight = []
    for _, row in labels.iterrows():
        if classFeature in row:
            if row[classFeature] != maskValue:
                sampleWeight.append(classWeights[int(row[classFeature])])
            else:
                sampleWeight.append(1)
        else:
            sampleWeight.append(1)
    return np.array(sampleWeight)


def create_datasets(data, targets, train=[], test=[]):

    if (len(train) == 0):
        train = data.copy()

    train_ds = df_to_dataset(train, targets=targets)
    train_features = train.copy()
    train_labels = {}
    for feature in targets:
        if feature['name'] in train_features:
            train_labels[feature['name']] = train_features.pop(feature['name'])
    train_labels = pd.DataFrame(train_labels)

    numCategoricalTargets = 0
    categoricalTarget = None
    for target in targets:
        if target.type == 'categorical':
            categoricalTarget = target
            numCategoricalTargets += 1

    if numCategoricalTargets == 1:
        unique = pd.unique(data[categoricalTarget.name])

        counts = data[categoricalTarget.name].value_counts()
        numSamples = 0
        for c in categoricalTarget.classes:
            if c in counts:
                numSamples += counts[c]

        classWeights = []
        for c in categoricalTarget.classes:
            if c in counts:
                classWeights.append(numSamples / (2 * counts[c]))
            else:
                classWeights.append(1.0)

        sampleWeight = generate_sample_weights(
            train_labels, categoricalTarget.name, classWeights)
    else:
        sampleWeight = [1]*len(train_labels)

    if len(test) > 0:
        test_ds = df_to_dataset(test, targets=targets)
        test_features = test.copy()
        test_labels = {}
        for feature in targets:
            if feature['name'] in test_features:
                test_labels[feature['name']] = test_features.pop(
                    feature['name'])
        test_labels = pd.DataFrame(test_labels)

        if numCategoricalTargets == 1:
            sampleWeightTest = generate_sample_weights(
                test_labels, categoricalTarget.name, classWeights)
        else:
            sampleWeightTest = [1]*len(test_labels)

        return train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest
    else:
        return train_ds, train_features, train_labels, sampleWeight


def filter_masked(base, other=None):
    filtered_base = []
    filtered_other = []

    i = 0
    for _, value in base.iteritems():
        if value != maskValue and not np.isnan(value):
            filtered_base.append(value)
            if other is not None:
                if isinstance(other, pd.Series):
                    filtered_other.append(other.iloc[i])
                else:
                    filtered_other.append(other[i])

        i += 1

    filtered_base = np.array(filtered_base)
    if other is not None:
        filtered_other = np.array(filtered_other)

        return filtered_base, filtered_other
    else:
        return filtered_base
