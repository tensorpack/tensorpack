# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

from ..utils import logger
from ..utils import *

__all__ = ['regularize_cost', 'l2_regularizer', 'l1_regularizer']

@memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))

l2_regularizer = tf.contrib.layers.l2_regularizer
l1_regularizer = tf.contrib.layers.l1_regularizer

def regularize_cost(regex, func, name=None):
    """
    Apply a regularizer on every trainable variable matching the regex
    """
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    costs = []
    for p in params:
        para_name = p.name
        if re.search(regex, para_name):
            costs.append(func(p))
            _log_regularizer(para_name)
    if not costs:
        return 0
    return tf.add_n(costs, name=name)

