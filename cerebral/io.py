import os
import pandas as pd

import cerebral as cb
from . import features


def load_data(plot=True,
              dropCorrelatedFeatures=True,
              model=None, tmp=False,
              additionalFeatures=[],
              postprocess=None):

    data_directory = cb.conf.data.directory
    data_files = cb.conf.data.files

    if (not os.path.exists(data_directory+'calculated_features.csv')
            or model is not None or tmp):

        data = []
        for data_file in data_files:
            if '.csv' in data_file:
                rawData = pd.read_csv(data_directory + data_file)
            elif '.xls' in data_file:
                rawData = pd.read_excel(data_directory + data_file, 'CD')
                if 'deltaT' in cb.conf.target_names:
                    rawData = pd.concat([rawData, pd.read_excel(
                        data_directory + data_file, 'SLR')])

            rawData = rawData.loc[:, ~rawData.columns.str.contains('^Unnamed')]

            if 'composition' not in rawData:
                rawData = features.calculate_compositions(rawData)

            data.append(rawData)

        data = pd.concat(data, ignore_index=True)

        if model is None:
            data = features.calculate_features(
                data, plot=plot,
                dropCorrelatedFeatures=dropCorrelatedFeatures,
                additionalFeatures=additionalFeatures)

        else:
            modelInputs = []
            for inputLayer in model.inputs:
                modelInputs.append(inputLayer.name)

            data = features.calculate_features(
                data, requiredFeatures=modelInputs,
                plot=plot,
                dropCorrelatedFeatures=dropCorrelatedFeatures,
                additionalFeatures=additionalFeatures)

        data = data.fillna(features.maskValue)

        if not tmp:
            data.to_csv(data_directory+'calculated_features.csv')
    else:
        data = pd.read_csv(data_directory+'calculated_features.csv')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    if 'deltaT' not in cb.conf.target_names and 'deltaT' in data:
        data = data.drop('deltaT', axis='columns')

    if postprocess is not None:
        data = postprocess(data)

    return data
