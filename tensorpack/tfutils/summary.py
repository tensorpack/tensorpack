# -*- coding: UTF-8 -*-
# File: summary.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import six
import tensorflow as tf
import re
import io
from six.moves import range
from contextlib import contextmanager

from tensorflow.python.training import moving_averages

from ..utils import logger
from ..utils.develop import log_deprecated
from ..utils.argtools import graph_memoized
from ..utils.naming import MOVING_SUMMARY_OPS_KEY
from .tower import get_current_tower_context
from .symbolic_functions import rms

__all__ = ['create_scalar_summary', 'add_param_summary',
           'add_activation_summary', 'add_moving_summary']


def create_scalar_summary(name, v):
    """
    Args:
        name (str):
        v (float): scalar value
    Returns:
        tf.Summary: a tf.Summary object with name and simple scalar value v.
    """
    assert isinstance(name, six.string_types), type(name)
    v = float(v)
    s = tf.Summary()
    s.value.add(tag=name, simple_value=v)
    return s


def create_image_summary(name, val):
    """
    Args:
        name(str):
        val(np.ndarray): 4D tensor of NHWC. assume RGB if C==3.
            Can be either float or uint8. Range has to be [0,255].

    Returns:
        tf.Summary:
    """
    assert isinstance(name, six.string_types), type(name)
    n, h, w, c = val.shape
    val = val.astype('uint8')
    s = tf.Summary()
    for k in range(n):
        arr = val[k]
        if arr.shape[2] == 1:   # scipy doesn't accept (h,w,1)
            arr = arr[:, :, 0]
        tag = name if n == 1 else '{}/{}'.format(name, k)

        buf = io.BytesIO()
        # scipy assumes RGB
        scipy.misc.toimage(arr).save(buf, format='png')

        img = tf.Summary.Image()
        img.height = h
        img.width = w
        # 1 - grayscale 3 - RGB 4 - RGBA
        img.colorspace = c
        img.encoded_image_string = buf.getvalue()
        s.value.add(tag=tag, image=img)
    return s


def add_activation_summary(x, name=None):
    """
    Add summary for an activation tensor x.  If name is None, use x.name.

    Args:
        x (tf.Tensor): the tensor to summary.
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    ndim = x.get_shape().ndims
    if ndim < 2:
        logger.warn("Cannot summarize scalar activation {}".format(x.name))
        return
    if name is None:
        name = x.name
    with tf.name_scope('activation-summary'):
        tf.summary.histogram(name, x)
        tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(x))
        tf.summary.scalar(name + '-rms', rms(x))


def add_param_summary(*summary_lists):
    """
    Add summary Ops for all trainable variables matching the regex.

    Args:
        summary_lists (list): each is (regex, [list of summary type to perform]).
        Summary type can be 'mean', 'scalar', 'histogram', 'sparsity', 'rms'
    """
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return
    if len(summary_lists) == 1 and isinstance(summary_lists[0], list):
        log_deprecated(text="Use positional args to call add_param_summary() instead of a list.")
        summary_lists = summary_lists[0]

    def perform(var, action):
        ndim = var.get_shape().ndims
        name = var.name.replace(':0', '')
        if action == 'scalar':
            assert ndim == 0, "Scalar summary on high-dimension data. Maybe you want 'mean'?"
            tf.summary.scalar(name, var)
            return
        assert ndim > 0, "Cannot perform {} summary on scalar data".format(action)
        if action == 'histogram':
            tf.summary.histogram(name, var)
            return
        if action == 'sparsity':
            tf.summary.scalar(name + '-sparsity', tf.nn.zero_fraction(var))
            return
        if action == 'mean':
            tf.summary.scalar(name + '-mean', tf.reduce_mean(var))
            return
        if action == 'rms':
            tf.summary.scalar(name + '-rms', rms(var))
            return
        raise RuntimeError("Unknown summary type: {}".format(action))

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with tf.name_scope('param-summary'):
        for p in params:
            name = p.name
            for rgx, actions in summary_lists:
                if not rgx.endswith('$'):
                    rgx = rgx + '(:0)?$'
                if re.match(rgx, name):
                    for act in actions:
                        perform(p, act)


@graph_memoized
def _get_cached_vs(name):
    with tf.variable_scope(name) as scope:
        return scope


@contextmanager
def _enter_vs_reuse_ns(name):
    vs = _get_cached_vs(name)
    with tf.variable_scope(vs):
        with tf.name_scope(vs.original_name_scope):
            yield vs


def add_moving_summary(*args, **kwargs):
    """
    Enable moving average summary for some tensors.
    It's only effective in the main training tower, otherwise calling this
    function is a no-op.

    Args:
        args: tensors to summary
        decay (float): the decay rate. Defaults to 0.95.
        collection (str or None): the name of the collection to add EMA-maintaining ops.
            The default will work together with the default
            :class:`MovingAverageSummary` callback.

    Returns:
        [tf.Tensor]: list of tensors returned by assign_moving_average,
            which can be used to maintain the EMA.
    """
    decay = kwargs.pop('decay', 0.95)
    coll = kwargs.pop('collection', MOVING_SUMMARY_OPS_KEY)
    assert len(kwargs) == 0, "Unknown arguments: " + str(kwargs)

    ctx = get_current_tower_context()
    # allow ctx to be none
    if ctx is not None and not ctx.is_main_training_tower:
        return

    if not isinstance(args[0], list):
        v = args
    else:
        log_deprecated("Call add_moving_summary with positional args instead of a list!")
        v = args[0]
    for x in v:
        assert isinstance(x, tf.Tensor), x
        assert x.get_shape().ndims == 0, x.get_shape()
    G = tf.get_default_graph()
    # TODO variable not saved under distributed

    ema_ops = []
    for c in v:
        name = re.sub('tower[0-9]+/', '', c.op.name)
        with G.colocate_with(c), tf.name_scope(None):
            # assign_moving_average creates variables with op names, therefore clear ns first.
            with _enter_vs_reuse_ns('EMA') as vs:
                ema_var = tf.get_variable(name, shape=c.shape, dtype=c.dtype,
                                          initializer=tf.constant_initializer(), trainable=False)
                ns = vs.original_name_scope
            with tf.name_scope(ns):     # reuse VS&NS so that EMA_1 won't appear
                ema_op = moving_averages.assign_moving_average(
                    ema_var, c, decay,
                    zero_debias=True, name=name + '_EMA_apply')
            tf.summary.scalar(name + '-summary', ema_op)    # write the EMA value as a summary
            ema_ops.append(ema_op)
    if coll is not None:
        for op in ema_ops:
            # TODO a new collection to summary every step?
            tf.add_to_collection(coll, op)
    return ema_ops


try:
    import scipy.misc
except ImportError:
    from ..utils.develop import create_dummy_func
    create_image_summary = create_dummy_func('create_image_summary', 'scipy.misc')  # noqa
