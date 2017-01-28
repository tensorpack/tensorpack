#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dependency.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


""" Utilities to handle dependency """

__all__ = ['create_dummy_func', 'create_dummy_class']


def create_dummy_class(klass, dependency):
    """
    When a dependency of a class is not available, create a dummy class which throws ImportError when used.

    Args:
        klass (str): name of the class.
        dependency (str): name of the dependency.

    Returns:
        class: a class object
    """
    class _Dummy(object):
        def __init__(self, *args, **kwargs):
            raise ImportError("Cannot import '{}', therefore '{}' is not available".format(dependency, klass))
    return _Dummy


def create_dummy_func(func, dependency):
    """
    When a dependency of a function is not available, create a dummy function which throws ImportError when used.

    Args:
        func (str): name of the function.
        dependency (str): name of the dependency.

    Returns:
        function: a function object
    """
    def _dummy(*args, **kwargs):
        raise ImportError("Cannot import '{}', therefore '{}' is not available".format(dependency, func))
    return _dummy
