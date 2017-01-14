#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
from types import ModuleType
import six
import os
import os.path
# this line is necessary for _TFModuleFunc to work
import tensorflow as tf  # noqa: F401
from ..utils import logger

__all__ = ['LinearWrap']


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


for _, module_name, _ in walk_packages(
        [os.path.dirname(__file__)]):
    if not module_name.startswith('_'):
        _global_import(module_name)


class LinearWrap(object):
    """ A simple wrapper to easily create "linear" graph,
        consisting of layers / symbolic functions with only one input & output.
    """

    class _TFModuleFunc(object):
        def __init__(self, mod, tensor):
            self._mod = mod
            self._t = tensor

        def __getattr__(self, name):
            ret = getattr(self._mod, name)
            if isinstance(ret, ModuleType):
                return LinearWrap._TFModuleFunc(ret, self._t)
            else:
                # assume to be a tf function
                def f(*args, **kwargs):
                    o = ret(self._t, *args, **kwargs)
                    return LinearWrap(o)
                return f

    def __init__(self, tensor):
        """
        Args:
            tensor (tf.Tensor): the tensor to wrap
        """
        self._t = tensor

    def __getattr__(self, layer_name):
        layer = eval(layer_name)
        if hasattr(layer, 'f'):
            # this is a registered tensorpack layer
            # parse arguments by tensorpack model convention
            if layer.use_scope:
                def f(name, *args, **kwargs):
                    ret = layer(name, self._t, *args, **kwargs)
                    return LinearWrap(ret)
            else:
                def f(*args, **kwargs):
                    if len(args) and isinstance(args[0], six.string_types):
                        name, args = args[0], args[1:]
                        ret = layer(name, self._t, *args, **kwargs)
                    else:
                        ret = layer(self._t, *args, **kwargs)
                    return LinearWrap(ret)
            return f
        else:
            if layer_name != 'tf':
                logger.warn("You're calling LinearWrap.__getattr__ with something neither a layer nor 'tf'!")
            assert isinstance(layer, ModuleType)
            return LinearWrap._TFModuleFunc(layer, self._t)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function on the wrapped tensor.

        Returns:
            LinearWrap: ``LinearWrap(func(self.tensor(), *args, **kwargs))``.
        """
        ret = func(self._t, *args, **kwargs)
        return LinearWrap(ret)

    def __call__(self):
        """
        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._t

    def tensor(self):
        """
        Equivalent to ``self.__call__()``.

        Returns:
            tf.Tensor: the underlying wrapped tensor.
        """
        return self._t

    def print_tensor(self):
        """
        Print the underlying tensor and return self. Can be useful to get the
        name of tensors inside :class:`LinearWrap`.

        :return: self
        """
        print(self._t)
        return self
