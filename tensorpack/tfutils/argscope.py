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
def argscope(layers, **param):
    if not isinstance(layers, list):
        layers = [layers]

    def _check_args_exist(l):
        args = inspect.getargspec(l).args
        for k, v in six.iteritems(param):
            assert k in args, "No argument {} in {}".format(k, l.__name__)

    for l in layers:
        assert hasattr(l, 'f'), "{} is not a registered layer".format(l.__name__)
        _check_args_exist(l.f)

    new_scope = copy.copy(get_arg_scope())
    for l in layers:
        new_scope[l.__name__].update(param)
    _ArgScopeStack.append(new_scope)
    yield
    del _ArgScopeStack[-1]

def get_arg_scope():
    """
    :returns: the current argscope.
        An argscope is a dict of dict: dict[layername] = {arg: val}
    """
    if len(_ArgScopeStack) > 0:
        return _ArgScopeStack[-1]
    else:
        return defaultdict(dict)
