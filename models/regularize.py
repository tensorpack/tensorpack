#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re
from utils import logger

__all__ = ['regularize_cost']

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
            logger.info("Apply regularizer for {}".format(name))
            costs.append(func(p))
    return tf.add_n(costs)

