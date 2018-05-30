# -*- coding: utf-8 -*-
# File: argscope.py

from contextlib import contextmanager
from collections import defaultdict
import copy
from functools import wraps
from inspect import isfunction, getmembers

__all__ = ['argscope', 'get_arg_scope', 'enable_argscope_for_module']

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

    # def _check_args_exist(l):
    #     args = inspect.getargspec(l).args
    #     for k, v in six.iteritems(kwargs):
    #         assert k in args, "No argument {} in {}".format(k, l.__name__)

    for l in layers:
        assert hasattr(l, 'symbolic_function'), "{} is not a registered layer".format(l.__name__)
        # _check_args_exist(l.symbolic_function)

    new_scope = copy.copy(get_arg_scope())
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


def argscope_mapper(func):
    """Decorator for function to support argscope
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        actual_args = copy.copy(get_arg_scope()[func.__name__])
        actual_args.update(kwargs)
        out_tensor = func(*args, **actual_args)
        return out_tensor
    # argscope requires this property
    wrapped_func.symbolic_function = None
    return wrapped_func


def enable_argscope_for_module(module):
    """
    Overwrite all functions of a given module to support argscope.
    Note that this function monkey-patches the module and therefore could have unexpected consequences.
    It has been only tested to work well with `tf.layers` module.
    """
    for name, obj in getmembers(module):
        if isfunction(obj):
            setattr(module, name, argscope_mapper(obj))
