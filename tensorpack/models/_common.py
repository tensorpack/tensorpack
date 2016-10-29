# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from functools import wraps
import six
import copy, os

from ..tfutils.argscope import get_arg_scope
from ..tfutils.modelutils import get_shape_str
from ..tfutils.summary import add_activation_summary
from ..utils import logger

# make sure each layer is only logged once
_layer_logged = set()

def disable_layer_logging():
    class ContainEverything:
        def __contains__(self, x):
            return True
    # can use nonlocal in python3, but how
    globals()['_layer_logged'] = ContainEverything()

def layer_register(
        summary_activation=False,
        log_shape=True,
        use_scope=True):
    """
    Register a layer.
    :param summary_activation: Define the default behavior of whether to
        summary the output(activation) of this layer.
        Can be overriden when creating the layer.
    :param log_shape: log input/output shape of this layer
    """

    def wrapper(func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if use_scope:
                name, inputs = args[0], args[1]
                args = args[1:] # actual positional args used to call func
                assert isinstance(name, six.string_types), name
            else:
                assert not log_shape and not summary_activation
                inputs = args[0]
                name = None
            do_summary = kwargs.pop(
                'summary_activation', summary_activation)

            # TODO use inspect.getcallargs to enhance?
            # update from current argument scope
            actual_args = copy.copy(get_arg_scope()[func.__name__])
            actual_args.update(kwargs)

            if name is not None:
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
            else:
                # run the actual function
                outputs = func(*args, **actual_args)
            return outputs

        wrapped_func.f = func   # attribute to access the underlining function object
        wrapped_func.use_scope = use_scope
        return wrapped_func

    # need some special handling for sphinx to work with the arguments
    on_doc = os.environ.get('READTHEDOCS') == 'True' \
            or os.environ.get('TENSORPACK_DOC_BUILDING')
    if on_doc:
        from decorator import decorator
        wrapper = decorator(wrapper)

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
    # for use with tensorflow NHWC ops
    return [1] + shape2d(a) + [1]
