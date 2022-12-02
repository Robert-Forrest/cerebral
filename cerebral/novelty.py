from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

import cerebral as cb


def novelty(alloys, model, datafiles):
    """Calculate the Local-outlier-factor for a set of alloys with respect to a
    dataset"""

    data = cb.features.load_data(
        datafiles=datafiles,
        model=model,
        postprocess=cb.GFA.ensure_default_values_glass,
    )

    targets = cb.models.get_model_prediction_features(model)
    data = data.drop("composition", axis="columns")
    for target in targets:
        data = data.drop(target["name"], axis="columns")

    pca = PCA(n_components=3)
    pca_transformer = pca.fit(data)
    data_pca = pca_transformer.transform(data)

    lof = LocalOutlierFactor(novelty=True)
    lof.fit(data_pca)

    (
        predict_alloy_data,
        targets,
        input_features,
    ) = cb.features.calculate_features(
        alloys["alloy"],
        merge_duplicates=False,
        drop_correlated_features=False,
        model=model,
    )

    for col in predict_alloy_data.columns:
        if col not in data.columns:
            predict_alloy_data = predict_alloy_data.drop(col, axis=1)

    novelty = -lof.decision_function(
        pca_transformer.transform(predict_alloy_data)
    )

    return novelty
