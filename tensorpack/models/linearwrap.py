#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: linearwrap.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
from types import ModuleType
from .registry import get_registered_layer

__all__ = ['LinearWrap']


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
        layer = get_registered_layer(layer_name)
        if layer is not None:
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
            assert layer_name == 'tf', \
                "Calling LinearWrap.{}:" \
                " neither a layer nor 'tf'! " \
                "Did you forget to extract tensor from LinearWrap?".format(layer_name)
            import tensorflow as layer  # noqa
            assert isinstance(layer, ModuleType), layer
            return LinearWrap._TFModuleFunc(layer, self._t)

    def apply(self, func, *args, **kwargs):
        """
        Apply a function on the wrapped tensor.

        Returns:
            LinearWrap: ``LinearWrap(func(self.tensor(), *args, **kwargs))``.
        """
        ret = func(self._t, *args, **kwargs)
        return LinearWrap(ret)

    def apply2(self, func, *args, **kwargs):
        """
        Apply a function on the wrapped tensor. The tensor
        will be the second argument of func.

        Returns:
            LinearWrap: ``LinearWrap(func(args[0], self.tensor(), *args[1:], **kwargs))``.
        """
        ret = func(args[0], self._t, *(args[1:]), **kwargs)
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
