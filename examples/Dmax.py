import cerebral as cb

cb.setup(
    {
        "targets": [{"name": "Dmax", "loss": "Huber"}],
        "data": {"files": ["data.xls"]},
    }
)

data = cb.io.load_data(postprocess=cb.GFA.ensure_default_values_glass)

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
