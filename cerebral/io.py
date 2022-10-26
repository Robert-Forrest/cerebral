"""Module providing IO functionality."""

from typing import List
import os

import pandas as pd
import metallurgy as mg

import cerebral as cb


def load_data(
    plot: bool = False,
    drop_correlated_features: bool = True,
    model=None,
    tmp: bool = False,
    additional_features: List[str] = [],
    postprocess: bool = None,
) -> pd.DataFrame:
    """Load and process data for use by cerebral.

    :group: utils

    Parameters
    ----------

    plot
        If True, plot analytical graphs of the raw data.
    drop_correlated_features
        If True, cull pairs of correlated features.
    model
        Use an existing model to extract particular required input features.
    tmp
        If True, this data read will skip large tasks such as analytics or
        correlation culling.
    additional_features
        A list of additional feature names to calculate.
    postprocess
        A function to run on the data after loading.

    """

    data_directory = cb.conf.data.directory
    data_files = cb.conf.data.files

    if (
        not os.path.exists(data_directory + "calculated_features.csv")
        or model is not None
        or tmp
    ):

        data = []
        for data_file in data_files:
            if ".csv" in data_file:
                rawData = pd.read_csv(data_directory + data_file)
            elif ".xls" in data_file:
                rawData = pd.read_excel(data_directory + data_file)

                # rawData = pd.read_excel(data_directory + data_file, "CD")
                # if "deltaT" in cb.conf.target_names:
                #     rawData = pd.concat(
                #         [
                #             rawData,
                #             pd.read_excel(data_directory + data_file, "SLR"),
                #         ]
                #     )

            rawData = rawData.loc[:, ~rawData.columns.str.contains("^Unnamed")]

            if "composition" not in rawData:
                rawData = extract_compositions(rawData)

            data.append(rawData)

        data = pd.concat(data, ignore_index=True)

        if model is None:
            data = cb.features.calculate_features(
                data,
                plot=plot,
                drop_correlated_features=drop_correlated_features,
                additional_features=additional_features,
            )

        else:
            model_inputs = [input_layer.name for input_layer in model.inputs]

            data = cb.features.calculate_features(
                data,
                required_features=model_inputs,
                plot=plot,
                drop_correlated_features=drop_correlated_features,
                additional_features=additional_features,
            )

        data = data.fillna(cb.features.mask_value)

        if not tmp:
            data.to_csv(data_directory + "calculated_features.csv")
    else:
        data = pd.read_csv(data_directory + "calculated_features.csv")
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    if "deltaT" not in cb.conf.target_names and "deltaT" in data:
        data = data.drop("deltaT", axis="columns")

    if postprocess is not None:
        data = postprocess(data)

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
            if column not in cb.conf.target_names:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)
                if row[column] > 0:
                    composition[column] = row[column] / 100.0

        composition = mg.Alloy(composition, rescale=False)
        compositions.append(composition.to_string())

    data["composition"] = compositions
    for column in columns_to_drop:
        data = data.drop(column, axis="columns")

    return data
