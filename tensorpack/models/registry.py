# -*- coding: utf-8 -*-
# File: registry.py


import copy
import re
import collections
from functools import wraps
import six
import tensorflow as tf

from ..compat import tfv1
from ..tfutils.argscope import get_arg_scope
from ..tfutils.model_utils import get_shape_str
from ..utils import logger

# make sure each layer is only logged once
_LAYER_LOGGED = set()
_LAYER_REGISTRY = {}

__all__ = ['layer_register', 'disable_layer_logging']


_NameConflict = "LAYER_NAME_CONFLICT!!"


def _register(name, func):
    if name in _LAYER_REGISTRY:
        _LAYER_REGISTRY[name] = _NameConflict
        return
    if name in ['tf']:
        raise ValueError(logger.error("A layer cannot be named {}".format(name)))
    _LAYER_REGISTRY[name] = func

    # handle alias
    if name == 'Conv2DTranspose':
        _register('Deconv2D', func)


def get_registered_layer(name):
    """
    Args:
        name (str): the name of the layer, e.g. 'Conv2D'
    Returns:
        the wrapped layer function, or None if not registered.
    """
    ret = _LAYER_REGISTRY.get(name, None)
    if ret == _NameConflict:
        raise KeyError("Layer named '{}' is registered with `@layer_register` more than once!".format(name))
    return ret


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


class LayerShapeLogger():
    """
    A class that logs shapes of inputs/outputs of layers,
    during the possibly-nested calls to them.
    """
    def __init__(self):
        self.stack = collections.deque()
        self.depth = 0

    def _indent(self):
        return " " * (self.depth * 2)

    def push_inputs(self, name, message):
        while len(self.stack):
            item = self.stack.pop()
            logger.info(self._indent() + "'{}' input: {}".format(item[0], item[1]))
            self.depth += 1

        self.stack.append((name, message))

    def push_outputs(self, name, message):
        if len(self.stack):
            assert len(self.stack) == 1, self.stack
            assert self.stack[-1][0] == name, self.stack
            item = self.stack.pop()
            logger.info(self._indent() + "'{}': {} --> {}".format(name, item[1], message))
        else:
            self.depth -= 1
            logger.info(self._indent() + "'{}' output: {}".format(name, message))


_SHAPE_LOGGER = LayerShapeLogger()


def layer_register(
        log_shape=False,
        use_scope=True):
    """
    Args:
        log_shape (bool): log input/output shape of this layer
        use_scope (bool or None):
            Whether to call this layer with an extra first argument as variable scope.
            When set to None, it can be called either with or without
            the scope name argument, depend on whether the first argument
            is string or not.

    Returns:
        A decorator used to register a layer.

    Example:

    .. code-block:: python

        @layer_register(use_scope=True)
        def add10(x):
            return x + tf.get_variable('W', shape=[10])

        # use it:
        output = add10('layer_name', input)  # the function will be called under variable scope "layer_name".
    """

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            assert args[0] is not None, args
            if use_scope:
                name, inputs = args[0], args[1]
                args = args[1:]  # actual positional args used to call func
                assert isinstance(name, six.string_types), "First argument for \"{}\" should be a string. ".format(
                    func.__name__) + "Did you forget to specify the name of the layer?"
            else:
                assert not log_shape
                if isinstance(args[0], six.string_types):
                    if use_scope is False:
                        logger.warn(
                            "Please call layer {} without the first scope name argument, "
                            "or register the layer with use_scope=None to allow calling it "
                            "with scope names.".format(func.__name__))
                    name, inputs = args[0], args[1]
                    args = args[1:]  # actual positional args used to call func
                else:
                    inputs = args[0]
                    name = None
            if not (isinstance(inputs, (tf.Tensor, tf.Variable)) or
                    (isinstance(inputs, (list, tuple)) and
                        isinstance(inputs[0], (tf.Tensor, tf.Variable)))):
                raise ValueError("Invalid inputs to layer: " + str(inputs))

            # use kwargs from current argument scope
            actual_args = copy.copy(get_arg_scope()[func.__name__])
            # explicit kwargs overwrite argscope
            actual_args.update(kwargs)
            # if six.PY3:
            #     # explicit positional args also override argscope. only work in PY3
            #     posargmap = inspect.signature(func).bind_partial(*args).arguments
            #     for k in six.iterkeys(posargmap):
            #         if k in actual_args:
            #             del actual_args[k]

            if name is not None:        # use scope
                with tfv1.variable_scope(name) as scope:
                    # this name is only used to surpress logging, doesn't hurt to do some heuristics
                    scope_name = re.sub('tower[0-9]+/', '', scope.name)
                    do_log_shape = log_shape and scope_name not in _LAYER_LOGGED
                    if do_log_shape:
                        _SHAPE_LOGGER.push_inputs(scope.name, get_shape_str(inputs))

                    # run the actual function
                    outputs = func(*args, **actual_args)

                    if do_log_shape:
                        _SHAPE_LOGGER.push_outputs(scope.name, get_shape_str(outputs))
                        _LAYER_LOGGED.add(scope_name)
            else:
                # run the actual function
                outputs = func(*args, **actual_args)
            return outputs

        wrapped_func.use_scope = use_scope
        wrapped_func.__argscope_enabled__ = True
        _register(func.__name__, wrapped_func)
        return wrapped_func

    return wrapper
