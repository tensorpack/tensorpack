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
from .scope_utils import cached_name_scope

__all__ = ['add_tensor_summary', 'add_param_summary',
           'add_activation_summary', 'add_moving_summary']


# some scope stuff to use internally...
@graph_memoized
def _get_cached_vs(name):
    with tf.variable_scope(name) as scope:
        return scope


@contextmanager
def _enter_vs_reuse_ns(name):
    vs = _get_cached_vs(name)
    # XXX Not good to enter the cached vs directly, because this will clean-up custom getter
    # with tf.variable_scope(name, reuse=tf.AUTO_REUSE):    # available in 1.4 only
    with tf.variable_scope(vs):
        with tf.name_scope(vs.original_name_scope):
            yield vs


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


def add_tensor_summary(x, types, name=None, collections=None,
                       main_tower_only=True):
    """
    Summarize a tensor by different methods.

    Args:
        x (tf.Tensor): a tensor to summarize
        types (list[str]): summary types, can be scalar/histogram/sparsity/mean/rms
        name (str): summary name. Defaults to be the op name.
        collections (list[str]): collections of the summary ops.
        main_tower_only (bool): Only run under main training tower. If
            set to True, calling this function under other TowerContext
            has no effect.

    Examples:

    .. code-block:: python

        with tf.name_scope('mysummaries'):  # to not mess up tensorboard
            add_tensor_summary(
                tensor, ['histogram', 'rms', 'sparsity'], name='mytensor')
    """
    types = set(types)
    if name is None:
        name = x.op.name
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return

    SUMMARY_TYPES_DIC = {
        'scalar': lambda: tf.summary.scalar(name, x, collections=collections),
        'histogram': lambda: tf.summary.histogram(name, x, collections=collections),
        'sparsity': lambda: tf.summary.scalar(
            name + '-sparsity', tf.nn.zero_fraction(x),
            collections=collections),
        'mean': lambda: tf.summary.scalar(
            name + '-mean', tf.reduce_mean(x),
            collections=collections),
        'rms': lambda: tf.summary.scalar(
            name + '-rms', rms(x), collections=collections)
    }
    for typ in types:
        SUMMARY_TYPES_DIC[typ]()


def add_activation_summary(x, types=None, name=None, collections=None):
    """
    Call :func:`add_tensor_summary` under a reused 'activation-summary' name scope.
    This function is a no-op if not calling from main training tower.

    Args:
        x (tf.Tensor): the tensor to summary.
        types (list[str]): summary types, defaults to ``['sparsity', 'rms', 'histogram']``.
        name (str): if is None, use x.name.
        collections (list[str]): collections of the summary ops.
    """
    ndim = x.get_shape().ndims
    if ndim < 2:
        logger.warn("Cannot summarize scalar activation {}".format(x.name))
        return
    if types is None:
        types = ['sparsity', 'rms', 'histogram']
    with cached_name_scope('activation-summary'):
        add_tensor_summary(x, types, name=name, collections=collections)


def add_param_summary(*summary_lists, **kwargs):
    """
    Add summary ops for all trainable variables matching the regex, under a
    reused 'param-summary' name scope.
    This function is a no-op if not calling from main training tower.

    Args:
        summary_lists (list): each is (regex, [list of summary type]).
            Summary type is defined in :func:`add_tensor_summary`.
        collections (list[str]): collections of the summary ops.

    Examples:

    .. code-block:: python

        add_param_summary(
            ('.*/W', ['histogram', 'rms']),
            ('.*/gamma', ['scalar']),
        )
    """
    collections = kwargs.pop('collections', None)
    assert len(kwargs) == 0, "Unknown kwargs: " + str(kwargs)
    ctx = get_current_tower_context()
    if ctx is not None and not ctx.is_main_training_tower:
        return

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    with cached_name_scope('param-summary'):
        for p in params:
            name = p.op.name
            for rgx, actions in summary_lists:
                if not rgx.endswith('$'):
                    rgx = rgx + '$'
                if re.match(rgx, name):
                    add_tensor_summary(p, actions, name=name, collections=collections)


def add_moving_summary(*args, **kwargs):
    """
    Add moving average summary for some tensors.
    This function is a no-op if not calling from main training tower.

    Args:
        args: tensors to summarize
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
            if not c.dtype.is_floating:
                c = tf.cast(c, tf.float32)
            # assign_moving_average creates variables with op names, therefore clear ns first.
            with _enter_vs_reuse_ns('EMA') as vs:
                ema_var = tf.get_variable(name, shape=c.shape, dtype=c.dtype,
                                          initializer=tf.constant_initializer(), trainable=False)
                ns = vs.original_name_scope
            with tf.name_scope(ns):     # reuse VS&NS so that EMA_1 won't appear
                ema_op = moving_averages.assign_moving_average(
                    ema_var, c, decay,
                    zero_debias=True, name=name + '_EMA_apply')
            ema_ops.append(ema_op)
        with tf.name_scope(None):
            # cannot add it into colocate group -- will force everything to cpus
            tf.summary.scalar(name + '-summary', ema_op)    # write the EMA value as a summary
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
