#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
from types import ModuleType
import tensorflow as tf
import os
import os.path
from ..utils import logger

def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]

for _, module_name, _ in walk_packages(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_'):
        _global_import(module_name)


class LinearWrap(object):
    """ A simple wrapper to easily create linear graph,
        for layers with one input&output, or tf function with one input&output
    """

    class TFModuleFunc(object):
        def __init__(self, mod, tensor):
            self._mod = mod
            self._t = tensor

        def __getattr__(self, name):
            ret = getattr(self._mod, name)
            if isinstance(ret, ModuleType):
                return LinearWrap.TFModuleFunc(ret, self._t)
            else:
                # assume to be a tf function
                def f(*args, **kwargs):
                    o = ret(self._t, *args, **kwargs)
                    return LinearWrap(o)
                return f

    def __init__(self, tensor):
        self._t = tensor

    def __getattr__(self, layer_name):
        layer = eval(layer_name)
        if hasattr(layer, 'f'):
            # this is a registered tensorpack layer
            if layer.use_scope:
                def f(name, *args, **kwargs):
                    ret = layer(name, self._t, *args, **kwargs)
                    return LinearWrap(ret)
            else:
                def f(*args, **kwargs):
                    ret = layer(self._t, *args, **kwargs)
                    return LinearWrap(ret)
            return f
        else:
            if layer_name != 'tf':
                logger.warn("You're calling LinearWrap.__getattr__ with something neither a layer nor 'tf'!")
            assert isinstance(layer, ModuleType)
            return LinearWrap.TFModuleFunc(layer, self._t)

    def apply(self, func, *args, **kwargs):
        """ send tensor to the first argument of a simple func"""
        ret = func(self._t, *args, **kwargs)
        return LinearWrap(ret)

    def __call__(self):
        return self._t

    def tensor(self):
        return self._t

    def print_tensor(self):
        print(self._t)
        return self


