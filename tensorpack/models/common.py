# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from functools import wraps
import six
import copy

from ..tfutils.argscope import get_arg_scope
from ..tfutils.modelutils import get_shape_str
from ..tfutils.summary import add_activation_summary
from ..utils import logger
from ..utils.develop import building_rtfd

# make sure each layer is only logged once
_LAYER_LOGGED = set()
_LAYER_REGISTERED = {}

__all__ = ['layer_register', 'disable_layer_logging', 'get_registered_layer']


def _register(name, func):
    if name in _LAYER_REGISTERED:
        raise ValueError("Layer named {} is already registered!".format(name))
    if name in ['tf']:
        raise ValueError(logger.error("A layer cannot be named {}".format(name)))
    _LAYER_REGISTERED[name] = func


def get_registered_layer(name):
    """
    Args:
        name (str): the name of the layer, e.g. 'Conv2D'
    Returns:
        the wrapped layer function, or None if not registered.
    """
    return _LAYER_REGISTERED.get(name, None)


def disable_layer_logging():
    """
    Disable the shape logging for all layers from this moment on. Can be
    useful when creating multiple towers.
    """
    class ContainEverything:
        def __contains__(self, x):
            return True
    # can use nonlocal in python3, but how
    globals()['_LAYER_LOGGED'] = ContainEverything()


def layer_register(
        summary_activation=False,
        log_shape=True,
        use_scope=True):
    """
    Register a layer.

    Args:
        summary_activation (bool): Define the default behavior of whether to
            summary the output(activation) of this layer.
            Can be overriden when creating the layer.
        log_shape (bool): log input/output shape of this layer
        use_scope (bool): whether to call this layer with an extra first argument as scope.
            If set to False, will try to figure out whether the first argument
            is scope name or not.
    """

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if use_scope:
                name, inputs = args[0], args[1]
                args = args[1:]  # actual positional args used to call func
                assert isinstance(name, six.string_types), name
            else:
                assert not log_shape and not summary_activation
                if isinstance(args[0], six.string_types):
                    name, inputs = args[0], args[1]
                    args = args[1:]  # actual positional args used to call func
                else:
                    inputs = args[0]
                    name = None
            if not (isinstance(inputs, (tf.Tensor, tf.Variable)) or
                    (isinstance(inputs, (list, tuple)) and
                        isinstance(inputs[0], (tf.Tensor, tf.Variable)))):
                raise ValueError("Invalid inputs to layer: " + str(inputs))
            do_summary = kwargs.pop(
                'summary_activation', summary_activation)

            # TODO use inspect.getcallargs to enhance?
            # update from current argument scope
            actual_args = copy.copy(get_arg_scope()[func.__name__])
            actual_args.update(kwargs)

            if name is not None:
                with tf.variable_scope(name) as scope:
                    do_log_shape = log_shape and scope.name not in _LAYER_LOGGED
                    do_summary = do_summary and scope.name not in _LAYER_LOGGED
                    if do_log_shape:
                        logger.info("{} input: {}".format(scope.name, get_shape_str(inputs)))

                    # run the actual function
                    outputs = func(*args, **actual_args)

                    if do_log_shape:
                        # log shape info and add activation
                        logger.info("{} output: {}".format(
                            scope.name, get_shape_str(outputs)))
                        _LAYER_LOGGED.add(scope.name)

                    if do_summary:
                        if isinstance(outputs, list):
                            for x in outputs:
                                add_activation_summary(x, scope.name)
                        else:
                            add_activation_summary(outputs, scope.name)
            else:
                # run the actual function
                outputs = func(*args, **actual_args)
            return outputs

        wrapped_func.f = func   # attribute to access the underlying function object
        wrapped_func.use_scope = use_scope
        _register(func.__name__, wrapped_func)
        return wrapped_func

    # need some special handling for sphinx to work with the arguments
    if building_rtfd():
        from decorator import decorator
        wrapper = decorator(wrapper)

    return wrapper
