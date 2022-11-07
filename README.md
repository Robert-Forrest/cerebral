# cerebral

![Tests](https://github.com/Robert-Forrest/cerebral/actions/workflows/tests.yml/badge.svg)

Tool for creating multi-output deep ensemble neural-networks for alloy property modelling.

See our paper [Machine-learning improves understanding of glass formation in
metallic
systems](https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00026a) for
discussion of the model, it's architecture, and performance.

## Installation

The cerebral package can be installed from
[pypi](https://pypi.org/project/cerebral/) using pip:

``pip install cerebral``

Cerebral makes heavy use of the
[metallurgy](https://github.com/Robert-Forrest/metallurgy) package to manipulate
and approximate properties of alloys. Cerebral can be used with the
[evomatic](https://github.com/Robert-Forrest/evomatic) package to perform alloy
searching.

## Usage

Cerebral can be used to create multi-input mult-output deep neural networks for
the modelling of arbitrary alloy properties.

The following example shows configuration of cerebral to predict the "price"
property of an alloy, based on atomic percentages alone. Cerebral is configured
to load data for this problem from the tests directory - this data is for
demonstration and testing only, it is synthetically created by the
[metallurgy](https://github.com/Robert-Forrest/metallurgy) package for the Cu-Zr
binary alloy system.

```python
import cerebral as cb

cb.setup(
    {
        "targets": [{"name": "price"}],
        "input_features": [
            "percentages"
        ],
        "data": {"files": ["tests/CuZr_prices.csv"]},
    }
)

data = cb.features.load_data()
```

```
>>> data
     composition      price  Cu_percentage  Zr_percentage
0          Cu100   6.000000          1.000          0.000
1    Cu99.9Zr0.1   6.044626          0.999          0.001
2    Cu99.7Zr0.3   6.133763          0.997          0.003
3    Cu99.6Zr0.4   6.178273          0.996          0.004
4    Cu99.4Zr0.6   6.267177          0.994          0.006
..           ...        ...            ...            ...
662  Zr99.4Cu0.6  36.969779          0.006          0.994
663  Zr99.5Cu0.5  36.991515          0.005          0.995
664  Zr99.7Cu0.3  37.034949          0.003          0.997
665  Zr99.8Cu0.2  37.056646          0.002          0.998
666        Zr100  37.100000          0.000          1.000
```

Once a DataFrame of alloy compositions, input features, and prediction targets
is available, it can be used to train a model. The following example takes the
DataFrame created above, and trains a neural network to reproduce the target
features (for a maximum of 200 training epochs). The neural network model
produced is a standard Keras / TensorFlow model.

```python
model, history, train_data, test_data = cb.models.train_model(
    data, max_epochs=200
)

>>> model
<keras.engine.functional.Functional object at 0x7f1810feac80>

>>> history.history["loss"]
[22.522766767894105, 21.966949822959215, ...] 

```

Once a model has been created, cerebral provides automation for evaluating its
performance by comparison against the training and test datasets. Since the
pricing data is based on a very simple linear mixture, the model is able to
learn quite well the relationship between percentages of Cu and Zr and the
price.

```python   
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

>>> metrics
{
  'price': {
    'train': {
      'R_sq': 0.9994298579318788, 
      'RMSE': 0.21407108083268242, 
      'MAE': 0.16591635524599488
    }, 
    'test': {
      'R_sq': 0.9994089218056131,
      'RMSE': 0.21349478924250365, 
      'MAE': 0.1721696906690461
    }
  }
}

```

Futher, the model can be used to generate predictions for arbitrary alloys, as
long as the required input features are supplied. Here, we see that the simple
example model predicts price value for pure copper which is in the vicinity of
the value originally calculated by linear mixture: 

```python
cb.models.predict(model, "Cu100")["price"]
>>> {'price': array([6.60157898])} 

mg.calculate("Cu100", "price")
>>> 6.0
```

## Documentation

Documentation is available [here.](https://cerebral.readthedocs.io/en/latest/api.html)
