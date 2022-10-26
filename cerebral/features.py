"""Module providing feature processing functionality."""

from typing import List, Optional, Union, Tuple
from collections.abc import Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
import metallurgy as mg

import cerebral as cb


maskValue = -1

units = {
    "Dmax": "mm",
    "Tl": "K",
    "Tg": "K",
    "Tx": "K",
    "deltaT": "K",
    "price_linearmix": "\\$/kg",
    "price": "\\$/kg",
    "mixing_enthalpy": "kJ/mol",
    "mixing_Gibbs_free_energy": "kJ/mol",
}
inverse_units = {}


def setup_units():
    global inverse_units

    for feature in units:
        if "/" not in units[feature]:
            inverse_units[feature] = "1/" + units[feature]
        else:
            split_units = units[feature].split("/")
            inverse_units[feature] = split_units[1] + "/" + split_units[0]


def prettyName(feature_name: str) -> str:
    """Converts a a feature name string to a LaTeX formatted string

    :group: utils

    Parameters
    ----------

    feature_name
        The feature name to be formatted.
    """

    if cb.conf is not None:
        if feature_name in cb.conf.pretty_feature_names:
            return (
                r"$"
                + cb.conf.pretty_features[
                    cb.conf.pretty_feature_names.index(feature_name)
                ].pretty
                + "$"
            )

    name = ""
    featureParts = feature_name.split("_")
    if "linearmix" in feature_name or "deviation" in feature_name:
        if len(featureParts) > 1:
            if featureParts[-1] == "linearmix":
                name = r"$\Sigma$ "
            elif featureParts[-1] == "deviation":
                name = r"$\delta$ "
        name += " ".join(word.title() for word in featureParts[0:-1])
    else:
        name += " ".join(word.title() for word in featureParts)
    return name


def calculate_features(
    data: pd.DataFrame,
    drop_correlated_features: bool = True,
    plot: bool = False,
    additionalFeatures: List[str] = [],
    requiredFeatures: List[str] = [],
    merge_duplicates: bool = True,
    model: Optional = None,
):
    """Calculates features for a data set of alloy compositions.

    :group: utils

    Parameters
    ----------

    data
        The data set of alloy compositions.
    drop_correlated_features
        If True, pairs of correlated feautres will be culled.
    plot
        If True, graphs of the data set population will be created.
    additionalFeatures
        List of additional feature names to calculate.
    requiredFeatures
        List of required feature names to calculate.
    merge_duplicates
        If True, duplicate alloy compositions will be combined.
    model
        If provided, obtain feature names from existing model inputs.

    """

    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, Iterable) and not isinstance(
            data, (str, dict)
        ):
            data = [data]

        parsed_data = []
        for i in range(len(data)):
            alloy = data[i]
            if not isinstance(data[i], mg.Alloy):
                alloy = mg.Alloy(data[i], rescale=False)
            parsed_data.append(alloy.to_string())

        data = pd.DataFrame(parsed_data, columns=["composition"])

    if model is not None:
        drop_correlated_features = False
        merge_duplicates = False

        (
            input_features,
            target_names,
        ) = get_features_from_model(model)

    else:
        input_features = cb.conf.input_features
        target_names = cb.conf.target_names

    for additionalFeature in additionalFeatures:
        actual_feature = additionalFeature.split("_linearmix")[0].split(
            "_deviation"
        )[0]
        if (
            actual_feature not in input_features
            and actual_feature not in target_names
        ):
            input_features.append(actual_feature)

    if len(requiredFeatures) > 0:
        drop_correlated_features = False

        for feature in requiredFeatures:
            if feature in input_features:
                continue

            if "_linearmix" in feature:
                actual_feature = feature.split("_linearmix")[0]
                if actual_feature not in input_features:
                    input_features.append(actual_feature)

            elif "_deviation" in feature:
                actual_feature = feature.split("_deviation")[0]
                if actual_feature not in input_features:
                    input_features.append(actual_feature)

            else:
                input_features.append(feature)

    feature_values = {}

    for feature in input_features:
        if mg.get_property_function(feature) is None:
            feature_values[feature + "_linearmix"] = []
            feature_values[feature + "_deviation"] = []

            units[feature + "_deviation"] = "%"
        else:
            feature_values[feature] = []

    input_features = list(feature_values.keys())

    for column in data:
        if column == "composition":
            continue

        if not np.issubdtype(data[column].dtype, np.number):
            classes = data[column].unique()

            data[column] = data[column].map(
                {classes[i]: i for i in range(len(classes))}
            )
            data[column] = data[column].fillna(maskValue)
            data[column] = data[column].astype(np.int64)

    for _, row in data.iterrows():
        for feature in input_features:
            feature_values[feature].append(
                mg.calculate(row["composition"], feature)
            )

    for feature in input_features:
        data[feature] = feature_values[feature]

    data = data.fillna(maskValue)

    if merge_duplicates:
        data = merge_duplicate_compositions(data)

    if plot:
        cb.plots.plot_correlation(data)
        cb.plots.plot_feature_variation(data)

    if drop_correlated_features:

        data = drop_static_features(data, target_names)
        data = _drop_correlated_features(data, target_names)

    return data.copy()


def _drop_correlated_features(data, target_names):
    correlation = np.array(data.corr())

    correlatedDroppedFeatures = []
    for i in range(len(correlation) - 1):
        if (
            data.columns[i] not in correlatedDroppedFeatures
            and data.columns[i] not in target_names
            and data.columns[i] != "composition"
        ):
            for j in range(i + 1, len(correlation)):
                if (
                    data.columns[j] not in correlatedDroppedFeatures
                    and data.columns[j] not in target_names
                    and data.columns[j] != "composition"
                ):
                    if np.abs(correlation[i][j]) >= cb.conf.train.get(
                        "correlation_threshold", 0.8
                    ):

                        if sum(np.abs(correlation[i])) < sum(
                            np.abs(correlation[j])
                        ):
                            # print(
                            #     data.columns[j],
                            #     sum(np.abs(correlation[j])),
                            #     "beats",
                            #     data.columns[i],
                            #     sum(np.abs(correlation[i])),
                            # )
                            correlatedDroppedFeatures.append(data.columns[i])
                            break
                        else:
                            # print(
                            #     data.columns[i],
                            #     sum(np.abs(correlation[i])),
                            #     "beats",
                            #     data.columns[j],
                            #     sum(np.abs(correlation[j])),
                            # )
                            correlatedDroppedFeatures.append(data.columns[j])

    for feature in correlatedDroppedFeatures:
        if feature not in target_names:
            data = data.drop(feature, axis="columns")

    return data


def drop_static_features(
    data: pd.DataFrame, target_names: List[str] = []
) -> pd.DataFrame:
    """Drop static features by analysis of the quartile coefficient of
    dispersion. See Equation 7 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: utils

    Parameters
    ----------

    data
        Dataset of alloy compositions and properties.
    target_names
        Dictionary of prediction target names.

    """

    staticFeatures = []

    quartileDiffusions = {}
    for feature in data.columns:
        if feature == "composition":
            continue

        Q1 = np.percentile(data[feature], 25)
        Q3 = np.percentile(data[feature], 75)

        coefficient = 0
        if np.abs(Q1 + Q3) > 0:
            coefficient = np.abs((Q3 - Q1) / (Q3 + Q1))
        quartileDiffusions[feature] = coefficient

        if coefficient < 0.1:
            staticFeatures.append(feature)

    for feature in staticFeatures:
        if feature not in target_names:
            data = data.drop(feature, axis="columns")

    return data


def merge_duplicate_compositions(data: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate composition entries by either dropping exact copies, or
    averaging the data of compositions with multiple experimental values.

    :group: utils

    Parameters
    ----------

    data
        Dataset of alloy compositions and properties.

    """

    data = data.drop_duplicates()
    to_drop = []
    seen_compositions = []
    duplicate_compositions = {}
    for i, row in data.iterrows():
        alloy = mg.Alloy(row["composition"], rescale=False)
        composition = alloy.to_string()

        if abs(1 - sum(alloy.composition.values())) > 0.01:
            print("Invalid composition:", row["composition"], i)
            to_drop.append(i)

        elif composition in seen_compositions:
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
        composition = mg.Alloy(row["composition"], rescale=False).to_string()

        if composition in duplicate_compositions:
            to_drop.append(i)

    data = data.drop(to_drop)

    deduplicated_rows = []
    for composition in duplicate_compositions:

        averaged_features = {}
        num_contributions = {}
        for feature in duplicate_compositions[composition][0].keys():
            if feature != "composition":
                averaged_features[feature] = 0
                num_contributions[feature] = 0

        for i in range(len(duplicate_compositions[composition])):
            for feature in averaged_features:
                if duplicate_compositions[composition][i][
                    feature
                ] != maskValue and not pd.isnull(
                    duplicate_compositions[composition][i][feature]
                ):
                    averaged_features[feature] += duplicate_compositions[
                        composition
                    ][i][feature]
                    num_contributions[feature] += 1

        for feature in averaged_features:
            if num_contributions[feature] == 0:
                averaged_features[feature] = maskValue
            elif num_contributions[feature] > 1:
                averaged_features[feature] /= num_contributions[feature]

        averaged_features["composition"] = composition

        deduplicated_rows.append(pd.DataFrame(averaged_features, index=[0]))

    if len(deduplicated_rows) > 0:
        deduplicated_data = pd.concat(deduplicated_rows, ignore_index=True)
        data = pd.concat([data, deduplicated_data], ignore_index=True)
    return data


def get_features_from_model(model):
    """Get names of features and targets from an existing model.

    :group: utils

    Parameters
    ----------

    model
        The model to extract names from.

    """

    targets = cb.models.get_model_prediction_features(model)
    target_names = [target["name"] for target in targets]

    input_features = cb.models.get_model_input_features(model)

    return input_features, target_names


def train_test_split(
    data, train_percentage=0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test subsets, ensuring that similar
    compositions are grouped together. See Section 3.1 of
    https://doi.org/10.1016/j.actamat.2018.08.002, and Section 4.1 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    :group: utils

    Parameters
    ----------

    data
        The dataset of alloy compositions.
    train_percentage
        The proportion of data to be separated into the training set.

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

    numTraining = np.ceil(
        int(train_percentage * len(unique_composition_spaces))
    )

    trainingSet = []
    testSet = []

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    for i in range(len(shuffled_unique_compositions)):
        compositions = unique_composition_spaces[
            shuffled_unique_compositions[i]
        ]
        if i < numTraining:
            trainingSet.extend(compositions)
        else:
            testSet.extend(compositions)

    return pd.DataFrame(trainingSet), pd.DataFrame(testSet)


def df_to_dataset(dataframe: pd.DataFrame, targets: List[str] = []):
    """Convert a pandas dataframe to a tensorflow dataset

    :group: utils

    Parameters
    ----------

    dataframe
        The DataFrame to convert to a dataset.
    targets
        List of prediction targets to label the dataset.

    """

    dataframe = dataframe.copy()

    labelNames = []
    for feature in targets:
        if feature["name"] in dataframe.columns:
            labelNames.append(feature["name"])

    if len(labelNames) > 0:
        labels = pd.concat([dataframe.pop(x) for x in labelNames], axis=1)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    batch_size = 1024
    if cb.conf:
        if cb.conf.get("train", None) is not None:
            batch_size = cb.conf.train.get("batch_size", batch_size)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    ds = ds.cache()

    return ds


def generate_sample_weights(
    samples: pd.DataFrame, class_feature: str, class_weights: List[float]
) -> np.array:
    """Based on per-class weights, generate per-sample weights.

    :group: utils

    Parameters
    ----------

    samples
        DataFrame containing data to assign weights to.
    class_feature
        The feature defining the class to which a sample belongs.
    class_weights
        The per-class weightings.

    """

    sample_weights = []
    for _, row in samples.iterrows():
        if class_feature in row:
            if row[class_feature] != maskValue:
                sample_weights.append(class_weights[int(row[class_feature])])
            else:
                sample_weights.append(1)
        else:
            sample_weights.append(1)
    return np.array(sample_weights)


def create_datasets(
    data: pd.DataFrame,
    targets: List[str],
    train: Union[list, pd.DataFrame] = [],
    test: Union[list, pd.DataFrame] = [],
):
    """Separates the total data set of alloy compositions into training and
    test subsets.

    :group: utils

    Parameters
    ----------

    data
        The dataset of alloy compositions.
    targets
        The features to be modelled by the neural network.
    train
        If provided, a preselected subset of data to be used for training.
    test
        If provided, a preselected subset of data to be used for testing.

    """

    if len(train) == 0:
        train = data.copy()

    train_ds = df_to_dataset(train, targets=targets)
    train_features = train.copy()
    train_labels = {}
    for feature in targets:
        if feature["name"] in train_features:
            train_labels[feature["name"]] = train_features.pop(feature["name"])
    train_labels = pd.DataFrame(train_labels)

    numCategoricalTargets = 0
    categoricalTarget = None
    for target in targets:
        if target.type == "categorical":
            categoricalTarget = target
            numCategoricalTargets += 1

    if numCategoricalTargets == 1:
        counts = data[categoricalTarget.name].value_counts()
        numSamples = 0
        for c in categoricalTarget.classes:
            if c in counts:
                numSamples += counts[c]

        class_weights = []
        for c in categoricalTarget.classes:
            if c in counts:
                class_weights.append(numSamples / (2 * counts[c]))
            else:
                class_weights.append(1.0)

        sample_weights = generate_sample_weights(
            train_labels, categoricalTarget.name, class_weights
        )
    else:
        sample_weights = [1] * len(train_labels)

    if len(test) > 0:
        test_ds = df_to_dataset(test, targets=targets)
        test_features = test.copy()
        test_labels = {}
        for feature in targets:
            if feature["name"] in test_features:
                test_labels[feature["name"]] = test_features.pop(
                    feature["name"]
                )
        test_labels = pd.DataFrame(test_labels)

        if numCategoricalTargets == 1:
            sample_weights_test = generate_sample_weights(
                test_labels, categoricalTarget.name, class_weights
            )
        else:
            sample_weights_test = [1] * len(test_labels)

        return (
            train_ds,
            test_ds,
            train_features,
            test_features,
            train_labels,
            test_labels,
            sample_weights,
            sample_weights_test,
        )
    else:
        return train_ds, train_features, train_labels, sample_weights


def filter_masked(data: pd.DataFrame, other: Optional[pd.DataFrame] = None):
    """Filters out masked or NaN values from a dataframe

    :group: utils

    Parameters
    ----------

    data
        The dataset to be filtered.
    other
        Any other data to be selected from based on the filtering of data.

    """

    filtered_data = []
    filtered_other = []

    i = 0
    for _, value in data.iteritems():
        if value != maskValue and not np.isnan(value):
            filtered_data.append(value)
            if other is not None:
                if isinstance(other, pd.Series):
                    filtered_other.append(other.iloc[i])
                else:
                    filtered_other.append(other[i])

        i += 1

    filtered_data = np.array(filtered_data)
    if other is not None:
        filtered_other = np.array(filtered_other)

        return filtered_data, filtered_other
    else:
        return filtered_data
