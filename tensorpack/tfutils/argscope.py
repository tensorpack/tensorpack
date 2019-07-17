# -*- coding: utf-8 -*-
# File: argscope.py

import copy
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import getmembers, isfunction
import tensorflow as tf

from ..compat import is_tfv2
from ..utils import logger
from .model_utils import get_shape_str
from .tower import get_current_tower_context

__all__ = ['argscope', 'get_arg_scope', 'enable_argscope_for_module',
           'enable_argscope_for_function']

_ArgScopeStack = []


@contextmanager
def argscope(layers, **kwargs):
    """
    Args:
        layers (list or layer): layer or list of layers to apply the arguments.

    Returns:
        a context where all appearance of these layer will by default have the
        arguments specified by kwargs.

    Example:
        .. code-block:: python

            with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, out_channel=32):
                x = Conv2D('conv0', x)
                x = Conv2D('conv1', x)
                x = Conv2D('conv2', x, out_channel=64)  # override argscope

    """
    if not isinstance(layers, list):
        layers = [layers]

    for l in layers:
        assert hasattr(l, '__argscope_enabled__'), "Argscope not supported for {}".format(l)

    # need to deepcopy so that changes to new_scope does not affect outer scope
    new_scope = copy.deepcopy(get_arg_scope())
    for l in layers:
        new_scope[l.__name__].update(kwargs)
    _ArgScopeStack.append(new_scope)
    yield
    del _ArgScopeStack[-1]


def get_arg_scope():
    """
    Returns:
        dict: the current argscope.

    An argscope is a dict of dict: ``dict[layername] = {arg: val}``
    """
    if len(_ArgScopeStack) > 0:
        return _ArgScopeStack[-1]
    else:
        return defaultdict(dict)


def enable_argscope_for_function(func, log_shape=True):
    """Decorator for function to support argscope

    Example:

        .. code-block:: python

            from mylib import myfunc
            myfunc = enable_argscope_for_function(myfunc)

    Args:
        func: A function mapping one or multiple tensors to one or multiple
            tensors.
        log_shape (bool): Specify whether the first input resp. output tensor
            shape should be printed once.

    Remarks:
        If the function ``func`` returns multiple input or output tensors,
        only the first input/output tensor shape is displayed during logging.

    Returns:
        The decorated function.

    """

    assert callable(func), "func should be a callable"

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        actual_args = copy.copy(get_arg_scope()[func.__name__])
        actual_args.update(kwargs)
        out_tensor = func(*args, **actual_args)
        in_tensor = args[0]

        ctx = get_current_tower_context()
        name = func.__name__ if 'name' not in kwargs else kwargs['name']
        if log_shape:
            if ('tower' not in ctx.ns_name.lower()) or ctx.is_main_training_tower:
                # we assume the first parameter is the most interesting
                if isinstance(out_tensor, tuple):
                    out_tensor_descr = out_tensor[0]
                else:
                    out_tensor_descr = out_tensor
                logger.info("{:<12}: {} --> {}".format(
                    "'" + name + "'",
                    get_shape_str(in_tensor),
                    get_shape_str(out_tensor_descr)))

        return out_tensor
    wrapped_func.__argscope_enabled__ = True
    return wrapped_func


def enable_argscope_for_module(module, log_shape=True):
    """
    Overwrite all functions of a given module to support argscope.
    Note that this function monkey-patches the module and therefore could
    have unexpected consequences.
    It has been only tested to work well with ``tf.layers`` module.

    Example:

        .. code-block:: python

            import tensorflow as tf
            enable_argscope_for_module(tf.layers)

    Args:
        log_shape (bool): print input/output shapes of each function.
    """
    if is_tfv2() and module == tf.layers:
        module = tf.compat.v1.layers
    for name, obj in getmembers(module):
        if isfunction(obj):
            setattr(module, name, enable_argscope_for_function(obj,
                    log_shape=log_shape))
