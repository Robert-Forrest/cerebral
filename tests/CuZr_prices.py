import cerebral as cb


def test_CuZr_price_model():

    cb.setup(
        {
            "targets": [
                {"name": "price"},
            ],
            "input_features": ["percentages"],
            "data": {"files": ["CuZr_prices.csv"]},
        }
    )

    data = cb.features.load_data()

    model, history, train_data, test_data = cb.models.train_model(
        data, max_epochs=200
    )

    (
        train_predictions,
        train_errors,
        test_predictions,
        test_errors,
        metrics,
    ) = cb.models.evaluate_model(
        model,
        train_data["dataset"],
        train_data["labels"],
        test_ds=test_data["dataset"],
        test_labels=test_data["labels"],
        train_compositions=train_data["compositions"],
        test_compositions=test_data["compositions"],
    )

    assert metrics["price"]["train"]["R_sq"] > 0.5
