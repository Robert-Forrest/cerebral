import cerebral as cb

cb.setup(
    {
        "targets": [{"name": "GFA", "type": "categorical"}, {"name": "Dmax"}],
        "input_features": [
            "melting_temperature",
            "series",
            "wigner_seitz_electron_density",
            "ideal_entropy",
            "mixing_enthalpy",
        ],
        "data": {"files": ["data.csv"]},
    }
)

data = cb.features.load_data(postprocess=cb.GFA.ensure_default_values_glass)

model, history, train_data, test_data = cb.models.train_model(data, plot=True)

train_predictions = cb.models.evaluate_model(
    model,
    train_data["dataset"],
    train_data["labels"],
    test_ds=test_data["dataset"],
    test_labels=test_data["labels"],
    train_compositions=train_data["compositions"],
    test_compositions=test_data["compositions"],
)
