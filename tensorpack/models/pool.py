#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy as np

from .common import layer_register
from ..utils.argtools import shape2d, shape4d
from ._test import TestModel


__all__ = ['MaxPooling', 'FixedUnPooling', 'AvgPooling', 'GlobalAvgPooling',
           'BilinearUpSample']


def _Pooling(func, x, shape, stride, padding, data_format):
    padding = padding.upper()
    shape = shape4d(shape, data_format=data_format)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride, data_format=data_format)

    return func(x, ksize=shape,
                strides=stride, padding=padding,
                data_format=data_format,
                name='output')


@layer_register()
def MaxPooling(x, shape, stride=None, padding='VALID', data_format='NHWC'):
    """
    Max Pooling on 4D tensors.

    Args:
        x (tf.Tensor): a 4D tensor.
        shape: int or (h, w) tuple
        stride: int or (h, w) tuple. Defaults to be the same as shape.
        padding (str): 'valid' or 'same'.

    Returns:
        tf.Tensor named ``output``.
    """
    return _Pooling(tf.nn.max_pool, x, shape, stride, padding,
                    data_format=data_format)


@layer_register()
def AvgPooling(x, shape, stride=None, padding='VALID', data_format='NHWC'):
    """
    Average Pooling on 4D tensors.

    Args:
        x (tf.Tensor): a 4D tensor.
        shape: int or (h, w) tuple
        stride: int or (h, w) tuple. Defaults to be the same as shape.
        padding (str): 'valid' or 'same'.

    Returns:
        tf.Tensor named ``output``.
    """
    return _Pooling(tf.nn.avg_pool, x, shape, stride, padding,
                    data_format=data_format)


@layer_register()
def GlobalAvgPooling(x, data_format='NHWC'):
    """
    Global average pooling as in the paper `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.

    Args:
        x (tf.Tensor): a NHWC tensor.
    Returns:
        tf.Tensor: a NC tensor named ``output``.
    """
    assert x.get_shape().ndims == 4
    assert data_format in ['NHWC', 'NCHW']
    axis = [1, 2] if data_format == 'NHWC' else [2, 3]
    return tf.reduce_mean(x, axis, name='output')


def UnPooling2x2ZeroFilled(x):
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        shv = tf.shape(x)
        ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
        ret.set_shape([None, None, None, sh[3]])
        return ret


@layer_register()
def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.

    Args:
        x (tf.Tensor): a NHWC tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.

    Returns:
        tf.Tensor: a NHWC tensor.
    """
    shape = shape2d(shape)

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
        return UnPooling2x2ZeroFilled(x)

    input_shape = tf.shape(x)
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.constant(mat, name='unpool_mat')
    elif isinstance(unpool_mat, np.ndarray):
        unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [-1])
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(tf.reshape(unpool_mat, [-1]), 0)  # 1x(shxsw)
    prod = tf.matmul(fx, mat)  # (bchw) x(shxsw)
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, tf.stack(
        [-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
    return prod


@layer_register()
def BilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.

    Args:
        x (tf.Tensor): a NHWC tensor
        shape (int): the upsample factor

    Returns:
        tf.Tensor: a NHWC tensor.
    """
    # inp_shape = tf.shape(x)
    # return tf.image.resize_bilinear(x,
    # tf.stack([inp_shape[1]*shape,inp_shape[2]*shape]),
    # align_corners=True)

    inp_shape = x.get_shape().as_list()
    ch = inp_shape[3]
    assert ch is not None

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        See https://github.com/BVLC/caffe/blob/master/include%2Fcaffe%2Ffiller.hpp#L244
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))
    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    deconv = tf.nn.conv2d_transpose(x, weight_var,
                                    tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32),
                                    [1, shape, shape, 1], 'SAME')

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)
    return deconv


class TestPool(TestModel):

    def test_fixed_unpooling(self):
        h, w = 3, 4
        mat = np.random.rand(h, w, 3).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 3])
        output = FixedUnPooling('unpool', inp, 2)
        res = self.run_variable(output)
        self.assertEqual(res.shape, (1, 2 * h, 2 * w, 3))

        # mat is on cornser
        ele = res[0, ::2, ::2, 0]
        self.assertTrue((ele == mat[:, :, 0]).all())
        # the rest are zeros
        res[0, ::2, ::2, :] = 0
        self.assertTrue((res == 0).all())

    def test_upsample(self):
        h, w = 5, 5
        scale = 2

        mat = np.random.rand(h, w).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 1])

        output = BilinearUpSample('upsample', inp, scale)
        res = self.run_variable(output)[0, :, :, 0]

        from skimage.transform import rescale
        res2 = rescale(mat, scale)

        diff = np.abs(res2 - res)

        # not equivalent to rescale on edge?
        diff[0, :] = 0
        diff[:, 0] = 0
        if not diff.max() < 1e-4:
            import IPython
            IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        self.assertTrue(diff.max() < 1e-4)
