# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from functools import wraps
import six
import copy
#from decorator import decorator

from ..tfutils import *
from ..tfutils.modelutils import *
from ..tfutils.summary import *
from ..utils import logger

# make sure each layer is only logged once
_layer_logged = set()

def layer_register(summary_activation=False, log_shape=True):
    """
    Register a layer.
    :param summary_activation: Define the default behavior of whether to
        summary the output(activation) of this layer.
        Can be overriden when creating the layer.
    :param log_shape: log input/output shape of this layer
    """
    #@decorator only enable me when building docs.
    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            name = args[0]
            assert isinstance(name, six.string_types), \
                    'name must be the first argument. Args: {}'.format(args)
            args = args[1:]
            do_summary = kwargs.pop(
                'summary_activation', summary_activation)
            inputs = args[0]

            # update from current argument scope
            actual_args = copy.copy(get_arg_scope()[func.__name__])
            actual_args.update(kwargs)

            with tf.variable_scope(name) as scope:
                do_log_shape = log_shape and scope.name not in _layer_logged
                do_summary = do_summary and scope.name not in _layer_logged
                if do_log_shape:
                    logger.info("{} input: {}".format(scope.name, get_shape_str(inputs)))

                # run the actual function
                outputs = func(*args, **actual_args)

                if do_log_shape:
                    # log shape info and add activation
                    logger.info("{} output: {}".format(
                        scope.name, get_shape_str(outputs)))
                    _layer_logged.add(scope.name)

                if do_summary:
                    if isinstance(outputs, list):
                        for x in outputs:
                            add_activation_summary(x, scope.name)
                    else:
                        add_activation_summary(outputs, scope.name)
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
