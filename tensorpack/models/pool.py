#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy

from ._common import *
from ..tfutils.symbolic_functions import *

__all__ = ['MaxPooling', 'FixedUnPooling', 'AvgPooling', 'GlobalAvgPooling']

@layer_register()
def MaxPooling(x, shape, stride=None, padding='VALID'):
    """
    MaxPooling on images.

    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor.
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)

@layer_register()
def AvgPooling(x, shape, stride=None, padding='VALID'):
    """
    Average pooling on images.

    :param input: NHWC tensor.
    :param shape: int or [h, w]
    :param stride: int or [h, w]. default to be shape.
    :param padding: 'valid' or 'same'. default to 'valid'
    :returns: NHWC tensor.
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.avg_pool(x, ksize=shape, strides=stride, padding=padding)

@layer_register()
def GlobalAvgPooling(x):
    """
    Global average pooling as in `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.

    :param input: NHWC tensor.
    :returns: NC tensor.
    """
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

@layer_register()
def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed mat to perform kronecker product with.

    :param input: NHWC tensor
    :param shape: int or [h, w]
    :param unpool_mat: a tf matrix with size=shape. If None, will use a mat
        with 1 at top-left corner.
    :returns: NHWC tensor
    """
    shape = shape2d(shape)
    input_shape = x.get_shape().as_list()
    assert len(input_shape) == 4
    if unpool_mat is None:
        mat = np.zeros(shape, dtype='float32')
        mat[0][0] = 1
        unpool_mat = tf.Variable(mat, trainable=False, name='unpool_mat')
    assert unpool_mat.get_shape().as_list() == list(shape)

    # perform a tensor-matrix kronecker product
    fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
    fx = tf.expand_dims(fx, -1)       # (bchw)x1
    mat = tf.expand_dims(flatten(unpool_mat), 0)    #1x(shxsw)
    prod = tf.matmul(fx, mat)    #(bchw) x(shxsw)
    prod = tf.reshape(prod, [-1, input_shape[3],
                             input_shape[1], input_shape[2],
                             shape[0], shape[1]])
    prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
    prod = tf.reshape(prod, [-1, input_shape[1] * shape[0],
                            input_shape[2] * shape[1],
                            input_shape[3]])
    return prod

from ._test import TestModel
class TestPool(TestModel):
    def test_fixed_unpooling(self):
        h, w = 3, 4
        mat = np.random.rand(h, w).astype('float32')
        inp = self.make_variable(mat)
        inp = tf.reshape(inp, [1, h, w, 1])
        output = FixedUnPooling('unpool', inp, 2)
        res = self.run_variable(output)
        self.assertEqual(res.shape, (1, 2*h, 2*w, 1))

        # mat is on cornser
        ele = res[0,::2,::2,0]
        self.assertTrue((ele == mat).all())
        # the rest are zeros
        res[0,::2,::2,0] = 0
        self.assertTrue((res == 0).all())
