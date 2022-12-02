"""Cerebral

A tool for creating multi-output deep ensemble neural networks with TensorFlow.
"""

import os

import metallurgy as mg
from omegaconf import OmegaConf

from . import features
from . import models
from . import metrics
from . import loss
from . import layers
from . import kfolds
from . import permutation
from . import plots
from . import tuning
from . import GFA
from .novelty import novelty

__all__ = [
    "features",
    "models",
    "layers",
    "loss",
    "metrics",
    "kfolds",
    "permutation",
    "plots",
    "tuning",
    "GFA",
    "novelty",
]

conf = OmegaConf.create({})


def setup(user_config: dict = {}):
    """Initialises cerebral's default parameters.

    :group: utils

    Parameters
    ----------

    user_config
        Parameters set by the user.

    """

    global conf

    conf = OmegaConf.create(user_config)

    if "targets" in conf:
        conf.target_names = [t.name for t in conf.targets]

        max_loss_weight = 0
        for i in range(len(conf.targets)):
            if "type" not in conf.targets[i]:
                conf.targets[i]["type"] = "numerical"
            if "weight" not in conf.targets[i]:
                conf.targets[i].weight = 1.0
            if "loss" not in conf.targets[i]:
                conf.targets[i].loss = "Huber"

            if conf.targets[i].weight > max_loss_weight:
                max_loss_weight = conf.targets[i].weight

        for i in range(len(conf.targets)):
            conf.targets[i].weight /= max_loss_weight

    else:
        raise Exception("No targets set!")

    if not hasattr(conf, "model_name"):
        conf.model_name = "_".join(conf.target_names)

    if not hasattr(conf, "plot"):
        conf.plot = OmegaConf.create({})
    if not hasattr(conf.plot, "model"):
        conf.plot.model = True
    if not hasattr(conf.plot, "features"):
        conf.plot.features = False

    if not hasattr(conf, "save"):
        conf.save = False

    if conf.save:
        if not hasattr(conf, "output_directory"):
            conf.output_directory = conf.model_name

    if hasattr(conf, "output_directory"):
        if conf.output_directory[-1] != "/":
            conf.output_directory += "/"

        if os.path.exists(conf.output_directory):
            raise FileExistsError(
                "Output directory: "
                + conf.output_directory
                + " already exists!"
            )
        os.makedirs(conf.output_directory)

    if not hasattr(conf, "data"):
        raise Exception("No data files set!")

    if "input_features" not in conf:
        conf.input_features = mg.get_all_properties()

    if "pretty_features" in conf:
        conf.pretty_feature_names = [f.name for f in conf.pretty_features]
    else:
        conf.pretty_features = []
        conf.pretty_feature_names = []

    if "train" not in conf:
        conf.train = OmegaConf.create({})
    if "max_epochs" not in conf.train:
        conf.train.max_epochs = 100
    if "train_percentage" not in conf.train:
        conf.train.train_percentage = 1.0

    features.setup_units()
