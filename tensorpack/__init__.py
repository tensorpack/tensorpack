# -*- coding: utf-8 -*-
# File: __init__.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy  # avoid https://github.com/tensorflow/tensorflow/issues/2034
import cv2  # avoid https://github.com/tensorflow/tensorflow/issues/1924

from tensorpack.train import *
from tensorpack.models import *
from tensorpack.utils import *
from tensorpack.tfutils import *
from tensorpack.callbacks import *
from tensorpack.dataflow import *
from tensorpack.predict import *

if int(numpy.__version__.split('.')[1]) < 9:
    logger.warn("Numpy < 1.9 could be extremely slow on some tasks.")

if get_tf_version() < 10:
    logger.error("tensorpack requires TensorFlow >= 0.10")
