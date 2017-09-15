#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy as np

from .shape_utils import StaticDynamicShape
from .common import layer_register
from ..utils.argtools import shape2d
from ._test import TestModel


__all__ = ['MaxPooling', 'FixedUnPooling', 'AvgPooling', 'GlobalAvgPooling',
           'BilinearUpSample']


@layer_register(log_shape=True)
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
    if stride is None:
        stride = shape
    ret = tf.layers.max_pooling2d(x, shape, stride, padding,
                                  'channels_last' if data_format == 'NHWC' else 'channels_first')
    return tf.identity(ret, name='output')


@layer_register(log_shape=True)
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
    if stride is None:
        stride = shape
    ret = tf.layers.average_pooling2d(x, shape, stride, padding,
                                      'channels_last' if data_format == 'NHWC' else 'channels_first')
    return tf.identity(ret, name='output')


@layer_register(log_shape=True)
def GlobalAvgPooling(x, data_format='NHWC'):
    """
    Global average pooling as in the paper `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.

    Args:
        x (tf.Tensor): a NHWC tensor.
    Returns:
        tf.Tensor: a NC tensor named ``output``.
    """
    assert x.shape.ndims == 4
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
        return ret


@layer_register(log_shape=True)
def FixedUnPooling(x, shape, unpool_mat=None, data_format='NHWC'):
    """
    Unpool the input with a fixed matrix to perform kronecker product with.

    Args:
        x (tf.Tensor): a 4D image tensor
        shape: int or (h, w) tuple
        unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
            If is None, will use a matrix with 1 at top-left corner.

    Returns:
        tf.Tensor: a 4D image tensor.
    """
    shape = shape2d(shape)

    output_shape = StaticDynamicShape(x)
    output_shape.apply(1 if data_format == 'NHWC' else 2, lambda x: x * shape[0])
    output_shape.apply(2 if data_format == 'NHWC' else 3, lambda x: x * shape[1])

    # a faster implementation for this special case
    if shape[0] == 2 and shape[1] == 2 and unpool_mat is None and data_format == 'NHWC':
        ret = UnPooling2x2ZeroFilled(x)
    else:
        # check unpool_mat
        if unpool_mat is None:
            mat = np.zeros(shape, dtype='float32')
            mat[0][0] = 1
            unpool_mat = tf.constant(mat, name='unpool_mat')
        elif isinstance(unpool_mat, np.ndarray):
            unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
        assert unpool_mat.shape.as_list() == list(shape)

        if data_format == 'NHWC':
            x = tf.transpose(x, [0, 3, 1, 2])
        # perform a tensor-matrix kronecker product
        x = tf.expand_dims(x, -1)       # bchwx1
        mat = tf.expand_dims(unpool_mat, 0)  # 1xshxsw
        ret = tf.tensordot(x, mat, axes=1)  # bxcxhxwxshxsw

        if data_format == 'NHWC':
            ret = tf.transpose(ret, [0, 2, 4, 3, 5, 1])
        else:
            ret = tf.transpose(ret, [0, 1, 2, 4, 3, 5])

        shape3_dyn = [output_shape.get_dynamic(k) for k in range(1, 4)]
        ret = tf.reshape(ret, tf.stack([-1] + shape3_dyn))

    ret.set_shape(tf.TensorShape(output_shape.get_static()))
    return ret


@layer_register(log_shape=True)
def BilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.

    Args:
        x (tf.Tensor): a NHWC tensor
        shape (int): the upsample factor

    Returns:
        tf.Tensor: a NHWC tensor.
    """
    inp_shape = x.shape.as_list()
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
    x = tf.pad(x, [[0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1], [0, 0]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, shape, shape, 1], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, shape, shape, 1], 'SAME')
    edge = shape * (shape - 1)
    deconv = deconv[:, edge:-edge, edge:-edge, :]

    if inp_shape[1]:
        inp_shape[1] *= shape
    if inp_shape[2]:
        inp_shape[2] *= shape
    deconv.set_shape(inp_shape)
    return deconv


class TestPool(TestModel):
    def test_FixedUnPooling(self):
        h, w = 3, 4
        scale = 2
        mat = np.random.rand(h, w, 3).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 3])
        output = FixedUnPooling('unpool', inp, scale)
        res = self.run_variable(output)
        self.assertEqual(res.shape, (1, scale * h, scale * w, 3))

        # mat is on cornser
        ele = res[0, ::scale, ::scale, 0]
        self.assertTrue((ele == mat[:, :, 0]).all())
        # the rest are zeros
        res[0, ::scale, ::scale, :] = 0
        self.assertTrue((res == 0).all())

    def test_BilinearUpSample(self):
        h, w = 12, 12
        scale = 2

        mat = np.random.rand(h, w).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 1])

        output = BilinearUpSample('upsample', inp, scale)
        res = self.run_variable(output)[0, :, :, 0]

        from skimage.transform import rescale
        res2 = rescale(mat, scale, mode='edge')

        diff = np.abs(res2 - res)

        # if not diff.max() < 1e-4:
        #     import IPython
        #     IPython.embed(config=IPython.terminal.ipapp.load_default_config())
        self.assertTrue(diff.max() < 1e-4, diff.max())
