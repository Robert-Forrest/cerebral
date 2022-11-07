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

        for i in range(len(conf.targets)):
            if "type" not in conf.targets[i]:
                conf.targets[i]["type"] = "numerical"
            if "weight" not in conf.targets[i]:
                conf.targets[i].weight = 1.0
            if "loss" not in conf.targets[i]:
                conf.targets[i].loss = "Huber"

    else:
        raise Exception("No targets set!")

    if not hasattr(conf, "model_name"):
        conf.model_name = "_".join(conf.target_names)

    if not hasattr(conf, "plot"):
        conf.plot = False

    if not hasattr(conf, "save"):
        conf.save = False

    if conf.save:
        if not hasattr(conf, "output_directory"):
            conf.output_directory = conf.model_name

    if hasattr(conf, "output_directory"):
        if not os.path.exists(conf.output_directory):
            os.makedirs(conf.output_directory)

        image_directory = conf.output_directory + "/figures"
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)
        image_directory = image_directory + "/"
        conf.image_directory = image_directory

    if not hasattr(conf, "data"):
        conf.data = OmegaConf.create({})

    if "directory" not in conf.data:
        conf.data.directory = "./"
    if conf.data.directory[-1] != "/":
        conf.data.directory += "/"

    if "files" not in conf.data:
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

    features.setup_units()
