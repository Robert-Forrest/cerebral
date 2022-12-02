"""Module providing feature processing functionality."""

from collections.abc import Iterable
from typing import List, Optional, Tuple, Union, Callable
from numbers import Number

import metallurgy as mg
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

import cerebral as cb

mask_value = -1

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


def load_data(
    datafiles: Optional[list] = None,
    drop_correlated_features: bool = True,
    model=None,
    postprocess: Callable = None,
    save_csv: bool = False,
) -> pd.DataFrame:
    """Load and process data for use by cerebral.

    :group: utils

    Parameters
    ----------

    datafiles
        List of data file paths to load from.
    drop_correlated_features
        If True, cull pairs of correlated features.
    model
        Use an existing model to extract particular required input features.
    postprocess
        A function to run on the data after loading.
    save_csv
        If True, save the calculated features as a csv file.

    """

    if datafiles is None:
        datafiles = cb.conf.get("data", None)
    if datafiles is None or len(datafiles) == 0:
        raise ValueError("No datafiles to load!")

    data = []
    for data_file in datafiles:
        raw_data = None
        if ".csv" in data_file:
            raw_data = pd.read_csv(data_file)
        elif ".xls" in data_file:
            raw_data = pd.read_excel(data_file)

        if raw_data is not None:

            raw_data = raw_data.loc[
                :, ~raw_data.columns.str.contains("^Unnamed")
            ]

            if "composition" not in raw_data:
                raw_data = extract_compositions(raw_data)

            data.append(raw_data)
        else:
            raise NotImplementedError(
                data_file + " filetype not yet implemented."
            )
    data = pd.concat(data, ignore_index=True)

    data, targets, input_features = calculate_features(
        data,
        drop_correlated_features=drop_correlated_features,
        model=model,
    )

    data = data.fillna(cb.features.mask_value)

    if postprocess is not None:
        data = postprocess(data, targets, input_features)

    if save_csv:
        data.to_csv(cb.conf.output_directory + "calculated_features.csv")

    return data


def extract_compositions(data: pd.DataFrame) -> pd.DataFrame:
    """Extracts alloy compositions from data files formatted with columns per
    element.

    :group: utils

    Parameters
    ----------

    data
        The raw data in a DataFrame.

    """

    compositions = []
    columns_to_drop = []
    for _, row in data.iterrows():
        composition = {}
        for column in data.columns:
            if column in mg.periodic_table.elements:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)
                if isinstance(row[column], Number):
                    if row[column] > 0:
                        composition[column] = row[column] / 100.0

        compositions.append(mg.Alloy(composition, rescale=False))

    data["composition"] = compositions
    for column in columns_to_drop:
        data = data.drop(column, axis="columns")

    return data


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
    required_features: List[str] = [],
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
    required_features
        List of required feature names to calculate.
    merge_duplicates
        If True, duplicate alloy compositions will be combined.
    model
        If provided, obtain feature names from existing model inputs.

    """

    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, Iterable) or isinstance(data, (str, dict)):
            data = [data]

        alloys = []
        for alloy in data:
            if not isinstance(alloy, mg.Alloy):
                alloys.append(mg.Alloy(alloy, rescale=False))
            else:
                alloys.append(alloy)

        data = pd.DataFrame(
            alloys,
            columns=["composition"],
        )
    else:
        alloys = []
        for i, row in data.iterrows():
            if not isinstance(row["composition"], mg.Alloy):
                alloys.append(mg.Alloy(row["composition"], rescale=False))
            else:
                alloys.append(row["composition"])

        data["composition"] = alloys

    data = drop_invalid_compositions(data)

    if model is not None:
        drop_correlated_features = False
        (
            input_features,
            targets,
        ) = get_features_from_model(model)
        target_names = [target["name"] for target in targets]

    else:
        input_features = cb.conf.input_features
        targets = cb.conf.targets
        target_names = cb.conf.target_names

    data = drop_unwanted_inputs(data, input_features, target_names)

    if len(required_features) > 0:

        for feature in required_features:
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

    original_input_features = input_features[:]
    input_features = []

    for feature in original_input_features:
        if "_linearmix" in feature:
            input_features.append(feature)
        elif "_deviation" in feature:
            input_features.append(feature)
            units[feature] = "%"
        elif feature == "percentages":
            unique_elements = mg.analyse.find_unique_elements(
                data["composition"]
            )
            for element in unique_elements:
                if element + "_percentage" not in input_features:
                    input_features.append(element + "_percentage")

        elif (
            mg.get_property_function(feature) is None
            and "_percentage" not in feature
        ):
            input_features.append(feature + "_linearmix")
            input_features.append(feature + "_deviation")

            units[feature + "_deviation"] = "%"
        else:
            input_features.append(feature)

    for column in data:
        if column == "composition":
            continue

        if not np.issubdtype(data[column].dtype, np.number):
            unique_classes = data[column].unique()
            classes = []
            for c in unique_classes:
                if isinstance(c, str) or not np.isnan(c):
                    classes.append(c)

            if column in target_names:
                for i in range(len(targets)):
                    if targets[i]["name"] == column:
                        targets[i]["classes"] = classes

            data[column] = (
                data[column]
                .map({classes[i]: i for i in range(len(classes))})
                .fillna(mask_value)
                .astype(np.int64)
            )

    if merge_duplicates:
        data = merge_duplicate_compositions(data, targets, target_names)

    input_feature_values = {}
    for feature in input_features:
        if "_percentage" not in feature:
            input_feature_values[feature] = mg.calculate(
                data["composition"], feature
            )
        else:
            input_feature_values[feature] = []
            element = feature.split("_percentage")[0]
            for i, row in data.iterrows():
                if element in row["composition"].composition:
                    input_feature_values[feature].append(
                        row["composition"].composition[element]
                    )
                else:
                    input_feature_values[feature].append(0)

    data = pd.concat(
        [data, pd.DataFrame.from_dict(input_feature_values)],
        axis=1,
    )

    data = data.fillna(mask_value)

    if cb.conf.get("plot", False) and cb.conf.plot.get("features", False):
        cb.plots.plot_correlation(data)
        cb.plots.plot_feature_variation(data)

    if drop_correlated_features:
        data = drop_static_features(data, target_names, required_features)
        data = remove_correlated_features(
            data, target_names, required_features
        )

    return data, targets, input_features


def drop_unwanted_inputs(
    data: pd.DataFrame, input_features: List[str], target_names: List[str]
) -> pd.DataFrame:
    """Remove columns from the input DataFrame if they are not specified as an
    input feature or a target feature.

    :group: utils

    Parameters
    ----------

    data
        Data to have unwanted features removed from.
    input_features
        List of names of input features.
    target_names
        List of names of target features.

    """

    to_drop = []
    for column in data:
        if column == "composition":
            continue
        if column not in input_features and column not in target_names:
            to_drop.append(column)

    return data.drop(to_drop, axis="columns")


def drop_invalid_compositions(data: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid alloy compositions from the input DataFrame. Alloy
    compositions are be invalid if they have percentages which do not sum to
    100%.

    :group: utils

    Parameters
    ----------

    data
        Data to have invalid compositions removed from.

    """

    to_drop = []
    for i, row in data.iterrows():
        if abs(1 - row["composition"].total_percentage) > 0.01:
            to_drop.append(i)

    return data.drop(to_drop).reset_index(drop=True)


def remove_correlated_features(data, target_names, required_features):
    """Remove highly correlated features from the training data.

    :group: utils

    Parameters
    ----------

    data
        Data to have invalid compositions removed from.
    target_names
        List of names of target features.
    required_features
        List of names of input features which cannot be removed.

    """

    correlation = np.array(data.corr())

    correlated_dropped_features = []
    for i in range(len(correlation) - 1):
        if (
            data.columns[i] not in correlated_dropped_features
            and data.columns[i] not in target_names
            and data.columns[i] not in required_features
            and data.columns[i] != "composition"
            and "_percentage" not in data.columns[i]
        ):
            for j in range(i + 1, len(correlation)):
                if (
                    data.columns[j] not in correlated_dropped_features
                    and data.columns[j] not in target_names
                    and data.columns[j] not in required_features
                    and data.columns[j] != "composition"
                ):
                    if np.abs(correlation[i][j]) >= cb.conf.train.get(
                        "correlation_threshold", 0.8
                    ):

                        if sum(np.abs(correlation[i])) < sum(
                            np.abs(correlation[j])
                        ):

                            correlated_dropped_features.append(data.columns[i])
                            break

                        correlated_dropped_features.append(data.columns[j])

    for feature in correlated_dropped_features:
        if feature not in target_names and feature not in required_features:
            data = data.drop(feature, axis="columns")

    return data.reset_index(drop=True)


def drop_static_features(
    data: pd.DataFrame,
    target_names: List[str] = [],
    required_features: List[str] = [],
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

    static_features = []

    quartile_dispersions = {}
    for feature in data.columns:
        if (
            feature == "composition"
            or feature in target_names
            or feature in required_features
            or "_percentage" in feature
        ):
            continue

        Q1 = np.percentile(data[feature], 25)
        Q3 = np.percentile(data[feature], 75)

        coefficient = 0
        if np.abs(Q1 + Q3) > 0:
            coefficient = np.abs((Q3 - Q1) / (Q3 + Q1))
        quartile_dispersions[feature] = coefficient

        if coefficient < 0.1:
            static_features.append(feature)

    for feature in static_features:
        if feature not in target_names and feature not in required_features:
            data = data.drop(feature, axis="columns")

    return data.reset_index(drop=True)


def merge_duplicate_compositions(
    data: pd.DataFrame, targets: list, target_names: list
) -> pd.DataFrame:
    """Merge duplicate composition entries by either dropping exact copies, or
    averaging the data of compositions with multiple experimental values.

    :group: utils

    Parameters
    ----------

    data
        Dataset of alloy compositions and properties.
    targets
        List of prediction targets.
    target_names
        List of prediction target names.

    """

    data = data.drop_duplicates()
    to_drop = []
    seen_compositions = []
    duplicate_compositions = {}
    for i, row in data.iterrows():
        alloy = row["composition"]
        composition_str = alloy.to_string()

        if composition_str in seen_compositions:
            if composition_str not in duplicate_compositions:
                duplicate_compositions[alloy] = [
                    data.iloc[seen_compositions.index(composition_str)]
                ]
            duplicate_compositions[alloy].append(row)
            to_drop.append(i)
        seen_compositions.append(composition_str)

    data = data.drop(to_drop)

    to_drop = []
    for i, row in data.iterrows():
        if row["composition"].to_string() in duplicate_compositions:
            to_drop.append(i)

    data = data.drop(to_drop)

    deduplicated_rows = []
    for composition in duplicate_compositions:
        feature_values = {}
        for feature in duplicate_compositions[composition][0].keys():
            if feature != "composition":
                feature_values[feature] = []

        for i in range(len(duplicate_compositions[composition])):
            for feature in feature_values:
                if duplicate_compositions[composition][i][
                    feature
                ] != mask_value and not pd.isnull(
                    duplicate_compositions[composition][i][feature]
                ):
                    feature_values[feature].append(
                        duplicate_compositions[composition][i][feature]
                    )

        for feature in feature_values:
            if len(feature_values[feature]) == 0:
                feature_values[feature] = mask_value
                continue

            if feature in target_names:
                categorical_feature = False
                for target in targets:
                    if target["name"] == feature:
                        if target["type"] == "categorical":
                            categorical_feature = True
                        break
                if categorical_feature:
                    feature_values[feature] = np.max(feature_values[feature])
                    continue

            feature_values[feature] = np.mean(feature_values[feature])

        feature_values["composition"] = composition

        deduplicated_rows.append(pd.DataFrame(feature_values, index=[0]))

    if len(deduplicated_rows) > 0:
        deduplicated_data = pd.concat(deduplicated_rows, ignore_index=True)
        data = pd.concat([data, deduplicated_data], ignore_index=True)

    return data.reset_index(drop=True)


def get_features_from_model(model):
    """Get names of features and targets from an existing model.

    :group: utils

    Parameters
    ----------

    model
        The model to extract names from.

    """

    targets = cb.models.get_model_prediction_features(model)

    input_features = cb.models.get_model_input_features(model)

    return input_features, targets


def train_test_split(
    data, train_percentage=0.75
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and test subsets, ensuring that similar
    compositions are grouped together. See Section 3.1 of
    https://doi.org/10.1016/j.actamat.2018.08.002, and Section 4.1 of
    https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a.

    If there is a composition type which dominates the dataset (over 60% of the
    compositions), standard random splitting will be applied to avoid very small
    training or test sets.

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
        sorted_composition = sorted(row["composition"].elements)
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    proportions = {}
    for composition_space in unique_composition_spaces:
        proportions[composition_space] = len(
            unique_composition_spaces[composition_space]
        ) / len(data)

    if not np.any([proportions[p] > 0.6 for p in proportions]):
        numTraining = np.ceil(
            int(train_percentage * len(unique_composition_spaces))
        )

        training_set = []
        test_set = []

        shuffled_unique_compositions = list(unique_composition_spaces.keys())
        np.random.shuffle(shuffled_unique_compositions)

        for i in range(len(shuffled_unique_compositions)):
            compositions = unique_composition_spaces[
                shuffled_unique_compositions[i]
            ]
            if i < numTraining:
                training_set.extend(compositions)
            else:
                test_set.extend(compositions)

    else:
        training_set, test_set = model_selection.train_test_split(
            data, train_size=train_percentage
        )
    return pd.DataFrame(training_set), pd.DataFrame(test_set)


def df_to_dataset(
    dataframe: pd.DataFrame,
    targets: List[str] = [],
    weights: List[float] = [],
    shuffle=True,
):
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

    if "composition" in dataframe:
        dataframe["composition"] = dataframe["composition"].map(
            lambda r: r.to_string()
        )

    label_names = []
    for feature in targets:
        if feature["name"] in dataframe.columns:
            label_names.append(feature["name"])

    if len(label_names) > 0:
        label_values = {}
        for label in label_names:
            label_values[label] = dataframe.pop(label)

        if len(weights) > 0:
            dataset = tf.data.Dataset.from_tensor_slices(
                (dict(dataframe), label_values, weights)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (dict(dataframe), label_values)
            )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    batch_size = 256
    if cb.conf:
        if cb.conf.get("train", None) is not None:
            batch_size = cb.conf.train.get("batch_size", batch_size)

    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))

    dataset = dataset.batch(batch_size).prefetch(batch_size)

    return dataset


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
            if row[class_feature] != mask_value:
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

    train_features = train.copy()

    train_labels = {}
    for feature in targets:
        if feature["name"] in train_features:
            train_labels[feature["name"]] = train_features.pop(feature["name"])
    train_labels = pd.DataFrame(train_labels)

    num_categorical_targets = 0
    categorical_target = None
    for target in targets:
        if target.type == "categorical":
            categorical_target = target
            num_categorical_targets += 1

    if num_categorical_targets == 1:
        classes = data[categorical_target["name"]].unique()
        counts = data[categorical_target["name"]].value_counts()
        num_samples = sum(counts)

        class_weights = []
        for c in classes:
            if c != mask_value:
                class_weights.append(float(num_samples / (2 * counts[c])))
            else:
                class_weights.append(1.0)

        sample_weights = generate_sample_weights(
            train_labels, categorical_target["name"], class_weights
        )
    else:
        sample_weights = [1.0] * len(train_labels)

    train_ds = df_to_dataset(train, targets=targets, weights=sample_weights)

    if len(test) > 0:

        test_features = test.copy()
        test_labels = {}
        for feature in targets:
            if feature["name"] in test_features:
                test_labels[feature["name"]] = test_features.pop(
                    feature["name"]
                )
        test_labels = pd.DataFrame(test_labels)

        if num_categorical_targets == 1:
            sample_weights_test = generate_sample_weights(
                test_labels, categorical_target["name"], class_weights
            )
        else:
            sample_weights_test = [1] * len(test_labels)

        test_ds = df_to_dataset(
            test, targets=targets, weights=sample_weights_test
        )

        return (train_ds, test_ds)

    return train_ds


def filter_masked(
    data: Union[pd.Series, list],
    other: Optional[Union[list, pd.Series]] = None,
):
    """Filters out masked or NaN values from data.

    :group: utils

    Parameters
    ----------

    data
        The data to be filtered.
    other
        Any other data to be selected from based on the filtering of data.

    """

    filtered_data = []
    filtered_other = []

    if isinstance(data, pd.Series):
        for i, value in data.items():
            if value != mask_value and not np.isnan(value):
                filtered_data.append(value)
                if other is not None:
                    if isinstance(other, pd.Series):
                        filtered_other.append(other.iloc[i])
                    else:
                        filtered_other.append(other[i])
    else:
        for i, value in enumerate(data):
            if value != mask_value and not np.isnan(value):
                filtered_data.append(value)
                if other is not None:
                    if isinstance(other, pd.Series):
                        filtered_other.append(other.iloc[i])
                    else:
                        filtered_other.append(other[i])

    filtered_data = np.array(filtered_data)

    if other is not None:
        filtered_other = np.array(filtered_other)

        return filtered_data, filtered_other

    return filtered_data
