#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: develop.py
# Author: tensorpack contributors


""" Utilities for developers only.
These are not visible to users (not automatically imported). And should not
appeared in docs."""
import os
import functools
from datetime import datetime
import importlib
import types

from . import logger


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
        dependency (str or list[str]): name(s) of the dependency.

    Returns:
        function: a function object
    """
    if isinstance(dependency, (list, str)):
        dependency = ','.join(dependency)

    def _dummy(*args, **kwargs):
        raise ImportError("Cannot import '{}', therefore '{}' is not available".format(dependency, func))
    return _dummy


def building_rtfd():
    """
    Returns:
        bool: if tensorpack is being imported to generate docs now.
    """
    return os.environ.get('READTHEDOCS') == 'True' \
        or os.environ.get('TENSORPACK_DOC_BUILDING')


def log_deprecated(name="", text="", eos=""):
    """
    Log deprecation warning.

    Args:
        name (str): name of the deprecated item.
        text (str, optional): information about the deprecation.
        eos (str, optional): end of service date such as "YYYY-MM-DD".
    """
    assert name or text
    if eos:
        eos = "after " + datetime(*map(int, eos.split("-"))).strftime("%d %b")
    if name:
        if eos:
            warn_msg = "%s will be deprecated %s. %s" % (name, eos, text)
        else:
            warn_msg = "%s was deprecated. %s" % (name, text)
    else:
        warn_msg = text
        if eos:
            warn_msg += " Legacy period ends %s" % eos
    logger.warn("[Deprecated] " + warn_msg)


def deprecated(text="", eos=""):
    """
    Args:
        text, eos: same as :func:`log_deprecated`.

    Returns:
        a decorator which deprecates the function.

    Example:
        .. code-block:: python

            @deprecated("Explanation of what to do instead.", "2017-11-4")
            def foo(...):
                pass
    """

    def get_location():
        import inspect
        frame = inspect.currentframe()
        if frame:
            callstack = inspect.getouterframes(frame)[-1]
            return '%s:%i' % (callstack[1], callstack[2])
        else:
            stack = inspect.stack(0)
            entry = stack[2]
            return '%s:%i' % (entry[1], entry[2])

    def deprecated_inner(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            name = "{} [{}]".format(func.__name__, get_location())
            log_deprecated(name, text, eos)
            return func(*args, **kwargs)
        return new_func
    return deprecated_inner


# Copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
class LazyLoader(types.ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(LazyLoader, self).__init__(name)

    def _load(self):
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
