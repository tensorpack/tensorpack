# -*- coding: UTF-8 -*-
# File: modelutils.py
# Author: tensorpack contributors

import tensorflow as tf
from termcolor import colored

from ..utils import logger
from .summary import add_moving_summary
from .tower import get_current_tower_context

__all__ = ['describe_model', 'get_shape_str', 'apply_slim_collections']


def describe_model():
    """ Print a description of the current model parameters """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.info("No trainable variables in the graph!")
        return
    msg = [""]
    total = 0
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        msg.append("{}: shape={}, dim={}".format(
            v.name, shape.as_list(), ele))
    size_mb = total * 4 / 1024.0**2
    msg.append(colored(
        "Total #param={} ({:.02f} MB assuming all float32)".format(total, size_mb), 'cyan'))
    logger.info(colored("Model Parameters: ", 'cyan') + '\n'.join(msg))


def get_shape_str(tensors):
    """
    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str


def apply_slim_collections(cost):
    """
    Apply slim collections to the cost, including:

    1. adding the cost with the regularizers in ``tf.GraphKeys.REGULARIZATION_LOSSES``.
    2. make the cost depend on ``tf.GraphKeys.UPDATE_OPS``.

    Args:
        cost: a scalar tensor

    Return:
        a scalar tensor, the cost after applying the collections.
    """
    regulization_losses = set(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    if len(regulization_losses) > 0:
        logger.info("Applying REGULARIZATION_LOSSES on cost.")
        reg_loss = tf.add_n(list(regulization_losses), name="regularize_loss")
        cost = tf.add(reg_loss, cost, name='total_cost')
        add_moving_summary(reg_loss, cost)

    # As these batch-norm statistics quickly accumulate, there is no significant loss of accuracy
    # if only the main tower handles all batch-normalization updates, which are then shared across
    # the towers
    ctx = get_current_tower_context()
    if ctx is not None and ctx.is_main_training_tower:
        non_grad_updates = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        if non_grad_updates:
            logger.info("Applying UPDATE_OPS collection from the first tower on cost.")
            with tf.control_dependencies(non_grad_updates):
                cost = tf.identity(cost, name='cost_with_update')
    return cost
