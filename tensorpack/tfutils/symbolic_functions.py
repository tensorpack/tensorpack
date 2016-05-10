# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np

def one_hot(y, num_labels):
    """
    :param y: prediction. an Nx1 int tensor.
    :param num_labels: an int. number of output classes
    :returns: an NxC onehot matrix.
    """
    with tf.op_scope([y, num_labels], 'one_hot'):
        batch_size = tf.size(y)
        y = tf.expand_dims(y, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, y])
        onehot_labels = tf.sparse_to_dense(
            concated, tf.pack([batch_size, num_labels]), 1.0, 0.0)
        onehot_labels.set_shape([None, num_labels])
        return tf.cast(onehot_labels, tf.float32)

def prediction_incorrect(logits, label, topk=1):
    """
    :param logits: NxC
    :param label: N
    :returns: a float32 vector of length N with 0/1 values, 1 meaning incorrect prediction
    """
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)), tf.float32)

def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])

def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    total_dim = np.prod(x.get_shape()[1:].as_list())
    return tf.reshape(x, [-1, total_dim])

def logSoftmax(x):
    """
    Batch log softmax.
    :param x: NxC tensor.
    :returns: NxC tensor.
    """
    with tf.op_scope([x], 'logSoftmax'):
        z = x - tf.reduce_max(x, 1, keep_dims=True)
        logprob = z - tf.log(tf.reduce_sum(tf.exp(z), 1, keep_dims=True))
        return logprob


def class_balanced_binary_class_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss for binary classification,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    :param pred: size: b x ANYTHING. the predictions in [0,1].
    :param label: size: b x ANYTHING. the ground truth in {0,1}.
    :returns: class-balanced binary classification cross entropy loss
    """
    z = batch_flatten(pred)
    y = batch_flatten(label)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    total = tf.add(count_neg, count_pos)
    beta = tf.truediv(count_neg, total)

    eps = 1e-8
    loss_pos = tf.mul(-beta, tf.reduce_sum(tf.mul(tf.log(tf.abs(z) + eps), y), 1))
    loss_neg = tf.mul(1. - beta, tf.reduce_sum(tf.mul(tf.log(tf.abs(1. - z) + eps), 1. - y), 1))
    cost = tf.sub(loss_pos, loss_neg)
    cost = tf.reduce_mean(cost, name=name)
    return cost

