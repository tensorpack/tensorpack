# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

from ..utils import logger
from ..utils.argtools import graph_memoized
from ..tfutils.tower import get_current_tower_context
from .common import layer_register

__all__ = ['regularize_cost', 'l2_regularizer', 'l1_regularizer', 'Dropout']


@graph_memoized
def _log_regularizer(name):
    logger.info("Apply regularizer for {}".format(name))


l2_regularizer = tf.contrib.layers.l2_regularizer
l1_regularizer = tf.contrib.layers.l1_regularizer


def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on trainable variables matching the regex.
    In replicated mode, will only regularize variables within the current tower.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.

    Returns:
        tf.Tensor: the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
    """
    ctx = get_current_tower_context()
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    costs = []
    for p in params:
        para_name = p.name
        # in replicated mode, only regularize variables inside this tower
        if ctx.has_own_variables and ctx.vs_name and (not para_name.startswith(ctx.vs_name)):
            continue
        if re.search(regex, para_name):
            costs.append(func(p))
            _log_regularizer(para_name)
    if not costs:
        return tf.constant(0, dtype=tf.float32, name='empty_regularize_cost')
    return tf.add_n(costs, name=name)


@layer_register(log_shape=False, use_scope=False)
def Dropout(x, keep_prob=0.5, is_training=None, noise_shape=None):
    """
    Dropout layer as in the paper `Dropout: a Simple Way to Prevent
    Neural Networks from Overfitting <http://dl.acm.org/citation.cfm?id=2670313>`_.

    Args:
        keep_prob (float): the probability that each element is kept. It is only used
            when is_training=True.
        is_training (bool): If None, will use the current :class:`tensorpack.tfutils.TowerContext`
            to figure out.
        noise_shape: same as `tf.nn.dropout`.
    """
    if is_training is None:
        is_training = get_current_tower_context().is_training
    keep_prob = tf.constant(keep_prob if is_training else 1.0)
    return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape)
