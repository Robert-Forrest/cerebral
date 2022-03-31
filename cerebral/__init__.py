import os
import datetime

from omegaconf import OmegaConf

from . import io
from . import features
from . import models
from . import metrics
from . import kfolds

conf = None


def setup(config="config.yaml"):
    global conf

    conf = OmegaConf.load(config)

    if not os.path.exists('output'):
        os.makedirs('output')

    model_name = conf.get('model_name', conf.task)
    if conf.get("output_directory", None) is None:
        conf.output_directory = 'output/' + model_name

    if not os.path.exists(conf.output_directory):
        os.makedirs(conf.output_directory)
    elif conf.task in ['simple', 'kFolds', 'kFoldsEnsemble']:
        print("Error: Model already exists with name: " + model_name)
        exit()

    image_directory = conf.output_directory + '/figures'
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
    image_directory = image_directory + '/'
    conf.image_directory = image_directory

    if conf.data.directory is None:
        print("Error: No data directory set")
        exit()
    elif conf.data.directory[-1] != "/":
        conf.data.directory += "/"

    conf.target_names = [t.name for t in conf.targets]
    conf.pretty_feature_names = [f.name for f in conf.pretty_features]
