# -*- coding: utf-8 -*-
# File: __init__.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy # avoid https://github.com/tensorflow/tensorflow/issues/2034
import cv2  # avoid https://github.com/tensorflow/tensorflow/issues/1924

from . import models
from . import train
from . import utils
from . import tfutils
from . import callbacks
from . import dataflow

from .train import *
from .models import *
from .utils import *
from .tfutils import *
from .callbacks import *
from .dataflow import *
from .predict import *

if int(numpy.__version__.split('.')[1]) < 9:
    logger.warn("Numpy < 1.9 could be extremely slow on some tasks.")
