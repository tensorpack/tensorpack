#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: argscope.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from contextlib import contextmanager
from collections import defaultdict
import inspect
import copy
import six

__all__ = ['argscope', 'get_arg_scope']

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

    def _check_args_exist(l):
        args = inspect.getargspec(l).args
        for k, v in six.iteritems(kwargs):
            assert k in args, "No argument {} in {}".format(k, l.__name__)

    for l in layers:
        assert hasattr(l, 'symbolic_function'), "{} is not a registered layer".format(l.__name__)
        _check_args_exist(l.symbolic_function)

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
