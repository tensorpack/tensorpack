# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

from ..utils import logger
from ..utils.argtools import memoized
from ..tfutils.tower import get_current_tower_context
from .common import layer_register

__all__ = ['regularize_cost', 'l2_regularizer', 'l1_regularizer', 'Dropout']


@memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))


l2_regularizer = tf.contrib.layers.l2_regularizer
l1_regularizer = tf.contrib.layers.l1_regularizer


def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on every trainable variable matching the regex.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.

    Returns:
        tf.Tensor: the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
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


@layer_register(log_shape=False, use_scope=False)
def Dropout(x, keep_prob=0.5, is_training=None):
    """
    Dropout layer as in the paper `Dropout: a Simple Way to Prevent
    Neural Networks from Overfitting <http://dl.acm.org/citation.cfm?id=2670313>`_.

    Args:
        keep_prob (float): the probability that each element is kept. It is only used
            when is_training=True.
        is_training (bool): If None, will use the current :class:`tensorpack.tfutils.TowerContext`
            to figure out.
    """
    if is_training is None:
        is_training = get_current_tower_context().is_training
    keep_prob = tf.constant(keep_prob if is_training else 1.0)
    return tf.nn.dropout(x, keep_prob)
