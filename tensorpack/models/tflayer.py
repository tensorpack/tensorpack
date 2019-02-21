# -*- coding: utf-8 -*-
# File: tflayer.py

import functools
import six
import tensorflow as tf

from ..tfutils.common import get_tf_version_tuple
from ..tfutils.varreplace import custom_getter_scope
from ..utils.argtools import get_data_format

__all__ = []


def map_common_tfargs(kwargs):
    df = kwargs.pop('data_format', None)
    if df is not None:
        df = get_data_format(df, keras_mode=True)
        kwargs['data_format'] = df

    old_nl = kwargs.pop('nl', None)
    if old_nl is not None:
        kwargs['activation'] = lambda x, name=None: old_nl(x, name=name)

    if 'W_init' in kwargs:
        kwargs['kernel_initializer'] = kwargs.pop('W_init')

    if 'b_init' in kwargs:
        kwargs['bias_initializer'] = kwargs.pop('b_init')
    return kwargs


def convert_to_tflayer_args(args_names, name_mapping):
    """
    After applying this decorator:
    1. data_format becomes tf.layers style
    2. nl becomes activation
    3. initializers are renamed
    4. positional args are transformed to corresponding kwargs, according to args_names
    5. kwargs are mapped to tf.layers names if needed, by name_mapping
    """

    def decorator(func):
        @functools.wraps(func)
        def decorated_func(inputs, *args, **kwargs):
            kwargs = map_common_tfargs(kwargs)

            posarg_dic = {}
            assert len(args) <= len(args_names), \
                "Please use kwargs instead of positional args to call this model, " \
                "except for the following arguments: {}".format(', '.join(args_names))
            for pos_arg, name in zip(args, args_names):
                posarg_dic[name] = pos_arg

            ret = {}
            for name, arg in six.iteritems(kwargs):
                newname = name_mapping.get(name, None)
                if newname is not None:
                    assert newname not in kwargs, \
                        "Argument {} and {} conflicts!".format(name, newname)
                else:
                    newname = name
                ret[newname] = arg
            ret.update(posarg_dic)  # Let pos arg overwrite kw arg, for argscope to work

            return func(inputs, **ret)

        return decorated_func

    return decorator


def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}

    Returns:
        A context where the variables are renamed.
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)


def rename_tflayer_get_variable():
    """
    Rename all :func:`tf.get_variable` with rules that transforms tflayer style to tensorpack style.

    Returns:
        A context where the variables are renamed.

    Example:

    .. code-block:: python

        with rename_tflayer_get_variable():
            x = tf.layer.conv2d(input, 3, 3, name='conv0')
            # variables will be named 'conv0/W', 'conv0/b'
    """
    mapping = {
        'kernel': 'W',
        'bias': 'b',
        'moving_mean': 'mean/EMA',
        'moving_variance': 'variance/EMA',
    }
    return rename_get_variable(mapping)


def monkeypatch_tf_layers():
    if get_tf_version_tuple() < (1, 4):
        if not hasattr(tf.layers, 'Dense'):
            from tensorflow.python.layers.core import Dense
            tf.layers.Dense = Dense

            from tensorflow.python.layers.normalization import BatchNormalization
            tf.layers.BatchNormalization = BatchNormalization

            from tensorflow.python.layers.convolutional import Conv2DTranspose, Conv2D
            tf.layers.Conv2DTranspose = Conv2DTranspose
            tf.layers.Conv2D = Conv2D

            from tensorflow.python.layers.pooling import MaxPooling2D, AveragePooling2D
            tf.layers.MaxPooling2D = MaxPooling2D
            tf.layers.AveragePooling2D = AveragePooling2D


monkeypatch_tf_layers()
