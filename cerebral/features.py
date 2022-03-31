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

import elementy
import cerebral as cb
import metallurgy as mg

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
        composition = mg.Alloy(composition)
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

    if feature in cb.conf.pretty_feature_names:
        return r'$'+cb.conf.pretty_features[cb.conf.pretty_feature_names.index(feature)].pretty+'$'
    else:
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
        merge_duplicates=True):

    basicFeatures = cb.conf.basicFeatures
    complexFeatures = cb.conf.complexFeatures

    for additionalFeature in additionalFeatures:
        actualFeature = additionalFeature.split(
            "_linearmix")[0].split('_deviation')[0]
        if (actualFeature not in basicFeatures
            and actualFeature not in complexFeatures
                and actualFeature not in cb.conf.target_names):
            basicFeatures.append(actualFeature)

    if len(requiredFeatures) > 0:
        dropCorrelatedFeatures = False

        for feature in requiredFeatures:

            if "_linearmix" in feature:
                actualFeature = feature.split("_linearmix")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            elif "_deviation" in feature:
                actualFeature = feature.split("_deviation")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            else:
                if feature not in complexFeatures:
                    complexFeatures.append(feature)

    featureValues = {}
    complexFeatureValues = {}

    for feature in basicFeatures:
        featureValues[feature] = {
            'linearmix': [],
            'deviation': []
        }
        units[feature + '_deviation'] = "%"

    for feature in complexFeatures:
        complexFeatureValues[feature] = []

    for feature in cb.conf.targets:
        if(feature.type == 'categorical' and feature.name in data.columns):
            data[feature.name] = data[feature.name].map(
                {feature.classes[i]: i for i in range(len(feature.classes))}
            )
            data[feature.name] = data[feature.name].fillna(maskValue)
            data[feature.name] = data[feature.name].astype(np.int64)

    for i, row in data.iterrows():

        composition = mg.alloy.parse_composition(row['composition'])

        for feature in basicFeatures:

            if 'linearmix' in featureValues[feature]:
                featureValues[feature]['linearmix'].append(
                    mg.linear_mixture(composition, feature))

            if 'deviation' in featureValues[feature]:
                featureValues[feature]['deviation'].append(
                    mg.deviation(composition, feature))

        if 'theoreticalDensity' in complexFeatureValues:
            complexFeatureValues['theoreticalDensity'].append(
                mg.density.calculate_theoretical_density(composition))

        if 'sValence' in complexFeatureValues:
            complexFeatureValues['sValence'].append(
                mg.valence.calculate_valence_proportion(composition, 's'))
        if 'pValence' in complexFeatureValues:
            complexFeatureValues['pValence'].append(
                mg.valence.calculate_valence_proportion(composition, 'p'))
        if 'dValence' in complexFeatureValues:
            complexFeatureValues['dValence'].append(
                mg.valence.calculate_valence_proportion(composition, 'd'))
        if 'fValence' in complexFeatureValues:
            complexFeatureValues['fValence'].append(
                mg.valence.calculate_valence_proportion(composition, 'f'))

        if 'ideal_entropy' in complexFeatureValues:
            complexFeatureValues['ideal_entropy'].append(
                mg.entropy.calculate_ideal_entropy(composition))

        if 'ideal_entropy_Xia' in complexFeatureValues:
            complexFeatureValues['ideal_entropy_Xia'].append(
                mg.entropy.calculate_ideal_entropy_xia(composition))

        if 'mismatch_entropy' in complexFeatureValues:
            complexFeatureValues['mismatch_entropy'].append(
                mg.entropy.calculate_mismatch_entropy(composition))

        if 'mixing_entropy' in complexFeatureValues:
            complexFeatureValues['mixing_entropy'].append(
                mg.entropy.calculate_mixing_entropy(composition))

        if 'structure_deviation' in complexFeatureValues:
            complexFeatureValues['structure_deviation'].append(
                mg.structures.calculate_structure_mismatch(composition))

        if 'block_deviation' in complexFeatureValues:
            complexFeatureValues['block_deviation'].append(
                mg.deviation(composition, 'block'))

        if 'series_deviation' in complexFeatureValues:
            complexFeatureValues['series_deviation'].append(
                mg.deviation(composition, 'series'))

        if 'mixing_enthalpy' in complexFeatureValues:
            complexFeatureValues['mixing_enthalpy'].append(
                mg.enthalpy.calculate_mixing_enthalpy(composition))

        if 'price' in complexFeatureValues:
            complexFeatureValues['price'].append(
                mg.price.calculate_price(composition))

        if 'mixing_Gibbs_free_energy' in complexFeatureValues:
            complexFeatureValues['mixing_Gibbs_free_energy'].append(
                mg.enthalpy.calculate_mixing_Gibbs_free_energy(composition,
                                                               complexFeatureValues['mixing_enthalpy'][-1],
                                                               featureValues['melting_temperature']['linearmix'][-1],
                                                               complexFeatureValues['mixing_entropy'][-1]))

        if 'mismatch_PHS' in complexFeatureValues:
            complexFeatureValues['mismatch_PHS'].append(
                mg.enthalpy.calculate_mismatch_PHS(
                    composition,
                    complexFeatureValues['mixing_enthalpy'][-1],
                    complexFeatureValues['mismatch_entropy'][-1])
            )

        if 'mixing_PHS' in complexFeatureValues:
            complexFeatureValues['mismatch_PHS'].append(
                mg.enthalpy.calculate_mixing_PHS(
                    composition,
                    complexFeatureValues['mixing_enthalpy'][-1],
                    complexFeatureValues['mixing_entropy'][-1])
            )

        if 'PHSS' in complexFeatureValues:
            complexFeatureValues['PHSS'].append(
                mg.enthalpy.calculate_mixing_PHSS(
                    composition,
                    complexFeatureValues['mixing_enthalpy'][-1],
                    complexFeatureValues['mixing_entropy'][-1],
                    complexFeatureValues['mismatch_entropy'][-1])
            )

        if 'viscosity' in complexFeatureValues:
            complexFeatureValues['viscosity'].append(
                mg.viscosity.calculate_viscosity(composition, complexFeatureValues['mixing_enthalpy'][-1]))

        if 'radius_gamma' in complexFeatureValues:
            complexFeatureValues['radius_gamma'].append(
                mg.radii.calculate_radius_gamma(composition))

        if 'lattice_distortion' in complexFeatureValues:
            complexFeatureValues['lattice_distortion'].append(
                mg.radii.calculate_lattice_distortion(composition))

        if 'EsnPerVec' in complexFeatureValues:
            complexFeatureValues['EsnPerVec'].append(
                mg.ratios.shell_to_valence_electron_concentration(
                    row['composition'],
                    featureValues['period']['linearmix'][-1],
                    featureValues['valence_electrons']['linearmix'][-1])
            )

        if 'EsnPerMn' in complexFeatureValues:
            complexFeatureValues['EsnPerMn'].append(
                mg.ratios.shell_to_valence_electron_concentration(
                    row['composition'],
                    featureValues['period']['linearmix'][-1],
                    featureValues['mendeleev_universal_sequence']['linearmix'][-1])
            )

        if 'thermodynamic_factor' in complexFeatureValues:
            complexFeatureValues['thermodynamic_factor'].append(
                mg.enthalpy.calculate_thermodynamic_factor(
                    featureValues['melting_temperature']['linearmix'][-1],
                    complexFeatureValues['mixing_entropy'][-1],
                    complexFeatureValues['mixing_enthalpy'][-1]
                )
            )

    for feature in featureValues:
        for kind in featureValues[feature]:
            if len(featureValues[feature][kind]) == len(data.index):
                data[feature + '_' + kind] = featureValues[feature][kind]
    for feature in complexFeatures:
        if len(complexFeatureValues[feature]) == len(data.index):
            data[feature] = complexFeatureValues[feature]

    data = data.drop_duplicates()
    data = data.fillna(maskValue)

    if merge_duplicates:
        to_drop = []
        seen_compositions = []
        duplicate_compositions = {}
        for i, row in data.iterrows():
            composition = mg.Alloy(row['composition']).to_string()

            if(not mg.alloy.valid_composition(row['composition'])):
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
            composition = mg.Alloy(row['composition']).to_string()

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
            if feature in cb.conf.target_names:
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
                and feature not in cb.conf.target_names
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


def df_to_dataset(dataframe):
    dataframe = dataframe.copy()

    labelNames = []
    for feature in cb.conf.targets:
        if feature.name in dataframe.columns:
            labelNames.append(feature.name)

    if len(labelNames) > 0:
        labels = pd.concat([dataframe.pop(x)
                            for x in labelNames], axis=1)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    batch_size = cb.conf.train.get('batch_size', 1024)

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


def create_datasets(data, train=[], test=[]):

    if (len(train) == 0):
        train = data.copy()

    train_ds = df_to_dataset(train)
    train_features = train.copy()
    train_labels = {}
    for feature in cb.conf.targets:
        if feature.name in train_features:
            train_labels[feature.name] = train_features.pop(feature.name)
    train_labels = pd.DataFrame(train_labels)

    numCategoricalTargets = 0
    categoricalTarget = None
    for target in cb.conf.targets:
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
        test_ds = df_to_dataset(test)
        test_features = test.copy()
        test_labels = {}
        for feature in cb.conf.targets:
            if feature.name in test_features:
                test_labels[feature.name] = test_features.pop(feature.name)
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
