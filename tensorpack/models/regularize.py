# -*- coding: utf-8 -*-
# File: regularize.py


import re
import tensorflow as tf

from ..compat import tfv1
from ..tfutils.common import get_tf_version_tuple
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import graph_memoized
from .common import layer_register

__all__ = ['regularize_cost', 'regularize_cost_from_collection',
           'l2_regularizer', 'l1_regularizer', 'Dropout']


@graph_memoized
def _log_once(msg):
    logger.info(msg)


if get_tf_version_tuple() <= (1, 12):
    l2_regularizer = tf.contrib.layers.l2_regularizer  # deprecated
    l1_regularizer = tf.contrib.layers.l1_regularizer  # deprecated
else:
    # oh these little dirty details
    l2_regularizer = lambda x: tf.keras.regularizers.l2(x * 0.5)  # noqa
    l1_regularizer = tf.keras.regularizers.l1


def regularize_cost(regex, func, name='regularize_cost'):
    """
    Apply a regularizer on trainable variables matching the regex, and print
    the matched variables (only print once in multi-tower training).
    In replicated mode, it will only regularize variables within the current tower.

    If called under a TowerContext with `is_training==False`, this function returns a zero constant tensor.

    Args:
        regex (str): a regex to match variable names, e.g. "conv.*/W"
        func: the regularization function, which takes a tensor and returns a scalar tensor.
            E.g., ``tf.nn.l2_loss, tf.contrib.layers.l1_regularizer(0.001)``.

    Returns:
        tf.Tensor: a scalar, the total regularization cost.

    Example:
        .. code-block:: python

            cost = cost + regularize_cost("fc.*/W", l2_regularizer(1e-5))
    """
    assert len(regex)
    ctx = get_current_tower_context()
    if not ctx.is_training:
        # Currently cannot build the wd_cost correctly at inference,
        # because ths vs_name used in inference can be '', therefore the
        # variable filter will fail
        return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # If vars are shared, regularize all of them
    # If vars are replicated, only regularize those in the current tower
    if ctx.has_own_variables:
        params = ctx.get_collection_in_tower(tfv1.GraphKeys.TRAINABLE_VARIABLES)
    else:
        params = tfv1.trainable_variables()

    names = []

    with tfv1.name_scope(name + '_internals'):
        costs = []
        for p in params:
            para_name = p.op.name
            if re.search(regex, para_name):
                regloss = func(p)
                assert regloss.dtype.is_floating, regloss
                # Some variables may not be fp32, but it should
                # be fine to assume regularization in fp32
                if regloss.dtype != tf.float32:
                    regloss = tf.cast(regloss, tf.float32)
                costs.append(regloss)
                names.append(p.name)
        if not costs:
            return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # remove tower prefix from names, and print
    if len(ctx.vs_name):
        prefix = ctx.vs_name + '/'
        prefixlen = len(prefix)

        def f(name):
            if name.startswith(prefix):
                return name[prefixlen:]
            return name
        names = list(map(f, names))
    logger.info("regularize_cost() found {} variables to regularize.".format(len(names)))
    _log_once("The following tensors will be regularized: {}".format(', '.join(names)))

    return tf.add_n(costs, name=name)


def regularize_cost_from_collection(name='regularize_cost'):
    """
    Get the cost from the regularizers in ``tf.GraphKeys.REGULARIZATION_LOSSES``.
    If in replicated mode, will only regularize variables created within the current tower.

    Args:
        name (str): the name of the returned tensor

    Returns:
        tf.Tensor: a scalar, the total regularization cost.
    """
    ctx = get_current_tower_context()
    if not ctx.is_training:
        # TODO Currently cannot build the wd_cost correctly at inference,
        # because ths vs_name used in inference can be '', therefore the
        # variable filter will fail
        return tf.constant(0, dtype=tf.float32, name='empty_' + name)

    # NOTE: this collection doesn't always grow with towers.
    # It only grows with actual variable creation, but not get_variable call.
    if ctx.has_own_variables:   # be careful of the first tower (name='')
        losses = ctx.get_collection_in_tower(tfv1.GraphKeys.REGULARIZATION_LOSSES)
    else:
        losses = tfv1.get_collection(tfv1.GraphKeys.REGULARIZATION_LOSSES)
    if len(losses) > 0:
        logger.info("regularize_cost_from_collection() found {} regularizers "
                    "in REGULARIZATION_LOSSES collection.".format(len(losses)))

        def maploss(l):
            assert l.dtype.is_floating, l
            if l.dtype != tf.float32:
                l = tf.cast(l, tf.float32)
            return l

        losses = [maploss(l) for l in losses]
        reg_loss = tf.add_n(losses, name=name)
        return reg_loss
    else:
        return tf.constant(0, dtype=tf.float32, name='empty_' + name)


@layer_register(use_scope=None)
def Dropout(x, *args, **kwargs):
    """
    Same as `tf.layers.dropout`.
    However, for historical reasons, the first positional argument is
    interpreted as keep_prob rather than drop_prob.
    Explicitly use `rate=` keyword arguments to ensure things are consistent.
    """
    if 'is_training' in kwargs:
        kwargs['training'] = kwargs.pop('is_training')
    if len(args) > 0:
        if args[0] != 0.5:
            logger.warn(
                "The first positional argument to tensorpack.Dropout is the probability to keep, rather than to drop. "
                "This is different from the rate argument in tf.layers.Dropout due to historical reasons. "
                "To mimic tf.layers.Dropout, explicitly use keyword argument 'rate' instead")
        rate = 1 - args[0]
    elif 'keep_prob' in kwargs:
        assert 'rate' not in kwargs, "Cannot set both keep_prob and rate!"
        rate = 1 - kwargs.pop('keep_prob')
    elif 'rate' in kwargs:
        rate = kwargs.pop('rate')
    else:
        rate = 0.5

    if kwargs.get('training', None) is None:
        kwargs['training'] = get_current_tower_context().is_training

    if get_tf_version_tuple() <= (1, 12):
        return tf.layers.dropout(x, rate=rate, **kwargs)
    else:
        return tf.nn.dropout(x, rate=rate if kwargs['training'] else 0.)
