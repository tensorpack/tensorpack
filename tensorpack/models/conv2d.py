# -*- coding: utf-8 -*-
# File: conv2d.py


from ..compat import tfv1 as tf  # this should be avoided first in model code

from ..tfutils.common import get_tf_version_tuple
from ..utils.argtools import get_data_format, shape2d, shape4d, log_once
from .common import VariableHolder, layer_register
from .tflayer import convert_to_tflayer_args, rename_get_variable

__all__ = ['Conv2D', 'Deconv2D', 'Conv2DTranspose']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2D(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        split=1):
    """
    Similar to `tf.layers.Conv2D`, but with some differences:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'.
    3. Support 'split' argument to do group convolution.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)  # deprecated
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')
    dilation_rate = shape2d(dilation_rate)

    if split == 1 and dilation_rate == [1, 1]:
        # tf.layers.Conv2D has bugs with dilations (https://github.com/tensorflow/tensorflow/issues/26797)
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2D(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias

    else:
        # group conv implementation
        data_format = get_data_format(data_format, keras_mode=False)
        in_shape = inputs.get_shape().as_list()
        channel_axis = 3 if data_format == 'NHWC' else 1
        in_channel = in_shape[channel_axis]
        assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
        assert in_channel % split == 0

        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Not supported by group conv or dilated conv!"

        out_channel = filters
        assert out_channel % split == 0
        assert dilation_rate == [1, 1] or get_tf_version_tuple() >= (1, 5), 'TF>=1.5 required for dilated conv.'

        kernel_shape = shape2d(kernel_size)
        filter_shape = kernel_shape + [in_channel / split, out_channel]
        stride = shape4d(strides, data_format=data_format)

        kwargs = {"data_format": data_format}
        if get_tf_version_tuple() >= (1, 5):
            kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

        # matching input dtype (ex. tf.float16) since the default dtype of variable if tf.float32
        inputs_dtype = inputs.dtype
        W = tf.get_variable(
            'W', filter_shape, dtype=inputs_dtype, initializer=kernel_initializer)

        if use_bias:
            b = tf.get_variable('b', [out_channel], dtype=inputs_dtype, initializer=bias_initializer)

        if split == 1:
            conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
        else:
            conv = None
            if get_tf_version_tuple() >= (1, 13):
                try:
                    conv = tf.nn.conv2d(inputs, W, stride, padding.upper(), **kwargs)
                except ValueError:
                    log_once("CUDNN group convolution support is only available with "
                             "https://github.com/tensorflow/tensorflow/pull/25818 . "
                             "Will fall back to a loop-based slow implementation instead!", 'warn')
            if conv is None:
                inputs = tf.split(inputs, split, channel_axis)
                kernels = tf.split(W, split, 3)
                outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

        ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
        if activation is not None:
            ret = activation(ret)
        ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b
    return ret


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['filters', 'kernel_size', 'strides'],
    name_mapping={
        'out_channel': 'filters',
        'kernel_shape': 'kernel_size',
        'stride': 'strides',
    })
def Conv2DTranspose(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Conv2DTranspose`.
    Some differences to maintain backward-compatibility:

    1. Default kernel initializer is variance_scaling_initializer(2.0).
    2. Default padding is 'same'

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    if kernel_initializer is None:
        if get_tf_version_tuple() <= (1, 12):
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(2.0)  # deprecated
        else:
            kernel_initializer = tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal')

    if get_tf_version_tuple() <= (1, 12):
        with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
            layer = tf.layers.Conv2DTranspose(
                filters,
                kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                _reuse=tf.get_variable_scope().reuse)
            ret = layer.apply(inputs, scope=tf.get_variable_scope())
            ret = tf.identity(ret, name='output')
        ret.variables = VariableHolder(W=layer.kernel)
        if use_bias:
            ret.variables.b = layer.bias
    else:
        # Our own implementation, to avoid Keras bugs. https://github.com/tensorflow/tensorflow/issues/25946
        assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
            "Unsupported arguments due to Keras bug in TensorFlow 1.13"
        data_format = get_data_format(data_format, keras_mode=False)
        shape_dyn = tf.shape(inputs)
        shape_sta = inputs.shape.as_list()
        strides2d = shape2d(strides)
        kernel_shape = shape2d(kernel_size)

        assert padding.lower() in ['valid', 'same'], "Padding {} is not supported!".format(padding)

        if padding.lower() == 'valid':
            shape_res2d = [max(kernel_shape[0] - strides2d[0], 0),
                           max(kernel_shape[1] - strides2d[1], 0)]
        else:
            shape_res2d = shape2d(0)

        if data_format == 'NCHW':
            channels_in = shape_sta[1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0], filters,
                 shape_dyn[2] * strides2d[0] + shape_res2d[0],
                 shape_dyn[3] * strides2d[1] + shape_res2d[1]])
            out_shape3_sta = [filters,
                              None if shape_sta[2] is None else shape_sta[2] * strides2d[0] + shape_res2d[0],
                              None if shape_sta[3] is None else shape_sta[3] * strides2d[1] + shape_res2d[1]]
        else:
            channels_in = shape_sta[-1]
            out_shape_dyn = tf.stack(
                [shape_dyn[0],
                 shape_dyn[1] * strides2d[0] + shape_res2d[0],
                 shape_dyn[2] * strides2d[1] + shape_res2d[1],
                 filters])
            out_shape3_sta = [None if shape_sta[1] is None else shape_sta[1] * strides2d[0] + shape_res2d[0],
                              None if shape_sta[2] is None else shape_sta[2] * strides2d[1] + shape_res2d[1],
                              filters]

        inputs_dtype = inputs.dtype
        W = tf.get_variable('W', kernel_shape + [filters, channels_in],
                            dtype=inputs_dtype, initializer=kernel_initializer)
        if use_bias:
            b = tf.get_variable('b', [filters], dtype=inputs_dtype, initializer=bias_initializer)
        conv = tf.nn.conv2d_transpose(
            inputs, W, out_shape_dyn,
            shape4d(strides, data_format=data_format),
            padding=padding.upper(),
            data_format=data_format)
        conv.set_shape(tf.TensorShape([shape_sta[0]] + out_shape3_sta))

        ret = tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv
        if activation is not None:
            ret = activation(ret)
        ret = tf.identity(ret, name='output')

        ret.variables = VariableHolder(W=W)
        if use_bias:
            ret.variables.b = b

    return ret


Deconv2D = Conv2DTranspose
