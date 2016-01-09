#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: pool.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import tensorflow as tf
import numpy

from ._common import *
from ..utils.symbolic_functions import *

__all__ = ['MaxPooling', 'FixedUnPooling']

@layer_register()
def MaxPooling(x, shape, stride=None, padding='VALID'):
    """
        shape, stride: int or list/tuple of length 2
        if stride is None, use shape by default
        padding: 'VALID' or 'SAME'
    """
    padding = padding.upper()
    shape = shape4d(shape)
    if stride is None:
        stride = shape
    else:
        stride = shape4d(stride)

    return tf.nn.max_pool(x, ksize=shape, strides=stride, padding=padding)


@layer_register()
def FixedUnPooling(x, shape, unpool_mat=None):
    """
    Unpool the input with a fixed mat to perform kronecker product with
    x: 4D tensor of (b, h, w, c)
    shape: int or list/tuple of length 2
    unpool_mat: a tf matrix with size=shape. if None, will use a zero-mat with a 1 as first element
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
