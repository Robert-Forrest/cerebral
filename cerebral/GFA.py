"""Module implementing functionality specific to GFA modelling"""

import numpy as np
import pandas as pd

import cerebral as cb


def ensure_default_values_glass(data: pd.DataFrame) -> pd.DataFrame:
    """Postprocessing function which assigns default values to compositions
    specific to GFA modellin

    :group: utils

    Parameters
    ----------

    data
        Data as read in by cb.features.load_data

    """

    for i, row in data.iterrows():

        try:
            _ = data.at[i, "Dmax"]
            hasDmax = True
        except BaseException:
            hasDmax = False

        if hasDmax:
            if not np.isnan(data.at[i, "Dmax"]):
                if row["Dmax"] == 0:
                    data.at[i, "GFA"] = 0
                elif row["Dmax"] <= 0.15:
                    data.at[i, "GFA"] = 1
                else:
                    data.at[i, "GFA"] = 2
            else:
                data.at[i, "Dmax"] = cb.features.mask_value
        else:
            data.at[i, "Dmax"] = cb.features.mask_value

        try:
            _ = data.at[i, "GFA"]
            hasGFA = True
        except BaseException:
            hasGFA = False

        if hasGFA:
            if not np.isnan(data.at[i, "GFA"]):
                if int(data.at[i, "GFA"]) == 0:
                    data.at[i, "Dmax"] = 0
                elif int(data.at[i, "GFA"]) == 1:
                    data.at[i, "Dmax"] = 0.15
                elif int(data.at[i, "GFA"]) == 2:
                    if "Dmax" in row:
                        if (
                            np.isnan(data.at[i, "Dmax"])
                            or data.at[i, "Dmax"] == 0
                            or data.at[i, "Dmax"] is None
                        ):
                            data.at[i, "Dmax"] = cb.features.mask_value
                    else:
                        data.at[i, "Dmax"] = cb.features.mask_value
                else:
                    data.at[i, "Dmax"] = cb.features.mask_value
            else:
                data.at[i, "GFA"] = cb.features.mask_value
        else:
            data.at[i, "GFA"] = cb.features.mask_value

        if "Tx" in row and "Tg" in row:
            if (
                not np.isnan(row["Tx"])
                and not np.isnan(row["Tg"])
                and not row["Tx"] == cb.features.mask_value
                and not row["Tg"] == cb.features.mask_value
            ):
                data.at[i, "deltaT"] = row["Tx"] - row["Tg"]
            else:
                data.at[i, "deltaT"] = cb.features.mask_value

    return data
