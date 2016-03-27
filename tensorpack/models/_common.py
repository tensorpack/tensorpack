# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from functools import wraps
import six

from ..tfutils.modelutils import *
from ..tfutils.summary import *
from ..utils import logger

# make sure each layer is only logged once
_layer_logged = set()

def layer_register(summary_activation=False):
    """
    Register a layer.
    Args:
        summary_activation:
            Define the default behavior of whether to
            summary the output(activation) of this layer.
            Can be overriden when creating the layer.
    """
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            name = args[0]
            assert isinstance(name, six.string_types), \
                    'name must be the first argument. Args: {}'.format(str(args))
            args = args[1:]

            do_summary = kwargs.pop(
                'summary_activation', summary_activation)
            inputs = args[0]
            with tf.variable_scope(name) as scope:
                outputs = func(*args, **kwargs)
                if scope.name not in _layer_logged:
                    # log shape info and add activation
                    logger.info("{} input: {}".format(
                        scope.name, get_shape_str(inputs)))
                    logger.info("{} output: {}".format(
                        scope.name, get_shape_str(outputs)))

                    if do_summary:
                        if isinstance(outputs, list):
                            for x in outputs:
                                add_activation_summary(x, scope.name)
                        else:
                            add_activation_summary(outputs, scope.name)
                    _layer_logged.add(scope.name)
                return outputs
        wrapped_func.f = func   # attribute to access the underlining function object
        return wrapped_func
    return wrapper

def shape2d(a):
    """
    a: a int or tuple/list of length 2
    """
    if type(a) == int:
        return [a, a]
    if isinstance(a, (list, tuple)):
        assert len(a) == 2
        return list(a)
    raise RuntimeError("Illegal shape: {}".format(a))

def shape4d(a):
    # for use with tensorflow
    return [1] + shape2d(a) + [1]
