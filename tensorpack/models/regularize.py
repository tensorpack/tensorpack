#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

from ..utils import logger
from ..utils import *

__all__ = ['regularize_cost']

@memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))

def regularize_cost(regex, func):
    """
    Apply a regularizer on every trainable variable matching the regex
    """
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    costs = []
    for p in params:
        name = p.name
        if re.search(regex, name):
            costs.append(func(p))
            _log_regularizer(name)
    return tf.add_n(costs)

