import metallurgy as mg
import cerebral as cb


def test_CuZr_price_model():

    cb.setup(
        {
            "targets": [
                {"name": "price"},
            ],
            "training": {"max_epochs": 1000},
            "input_features": ["percentages"],
            "data": ["tests/CuZr_prices.csv"],
            "plot": {"model": False, "data": False},
        }
    )

    data = cb.features.load_data()

    model, history, train_ds = cb.models.train_model(data, max_epochs=1000)

    (
        train_eval,
        metrics,
    ) = cb.models.evaluate_model(model, train_ds)

    assert metrics["price"]["train"]["R_sq"] > 0.5

    alloy = "Cu100"
    prediction = cb.models.predict(model, alloy)["price"][0]

    assert abs(prediction - mg.calculate(alloy, "price")) < 5.0
