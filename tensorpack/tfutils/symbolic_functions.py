# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
import numpy as np


def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
    """
    :param logits: NxC
    :param label: N
    :returns: a float32 vector of length N with 0/1 values. 1 means incorrect prediction
    """
    return tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, topk)),
                   tf.float32, name=name)


def flatten(x):
    """
    Flatten the tensor.
    """
    return tf.reshape(x, [-1])


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


def class_balanced_cross_entropy(pred, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    :param pred: size: b x ANYTHING. the predictions in [0,1].
    :param label: size: b x ANYTHING. the ground truth in {0,1}.
    :returns: class-balanced cross entropy loss
    """
    z = batch_flatten(pred)
    y = tf.cast(batch_flatten(label), tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    eps = 1e-12
    loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
    loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
    cost = tf.sub(loss_pos, loss_neg, name=name)
    return cost


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    This is more numerically stable than class_balanced_cross_entropy

    :param logits: size: the logits.
    :param label: size: the ground truth in {0,1}, of the same shape as logits.
    :returns: a scalar. class-balanced cross entropy loss
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)

    # logstable = tf.log(1 + tf.exp(-tf.abs(z)))
    # loss_pos = -beta * tf.reduce_mean(-y * (logstable - tf.minimum(0.0, z)))
    # loss_neg = (1. - beta) * tf.reduce_mean((y - 1.) * (logstable + tf.maximum(z, 0.0)))
    # cost = tf.sub(loss_pos, loss_neg, name=name)
    return cost


def print_stat(x, message=None):
    """ a simple print op.
        Use it like: x = print_stat(x)
    """
    if message is None:
        message = x.op.name
    return tf.Print(x, [tf.shape(x), tf.reduce_mean(x), x], summarize=20,
                    message=message, name='print_' + x.op.name)


def rms(x, name=None):
    if name is None:
        name = x.op.name + '/rms'
        with tf.name_scope(None):   # name already contains the scope
            return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)


def huber_loss(x, delta=1, name='huber_loss'):
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.reduce_sum(
        tf.select(abscost < delta,
                  sqrcost * 0.5,
                  abscost * delta - 0.5 * delta ** 2),
        name=name)


def get_scalar_var(name, init_value, summary=False, trainable=False):
    """
    get a scalar variable with certain initial value
    :param summary: summary this variable
    """
    ret = tf.get_variable(name, shape=[],
                          initializer=tf.constant_initializer(init_value),
                          trainable=trainable)
    if summary:
        # this is recognized in callbacks.StatHolder
        tf.summary.scalar(name + '-summary', ret)
    return ret


def saliency(output_op, input_op, name="saliency"):
    """Saliency image from network

    Parameters
    ----------
    output_op : TYPE
        start in network
    input_op : TYPE
        image-node in graph

    Returns
    -------
    TYPE
        saliency image with size of (input_op)
    """
    max_outp = tf.reduce_max(output_op, 1)
    saliency_op = tf.gradients(max_outp, input_op)[:][0]
    saliency_op = tf.identity(saliency_op, name=name)
    return saliency_op


def psnr_loss(prediction, ground_truth):
    """Peek Signal to Noise Ratio (negative)

    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    Assuming MAXp == 1, then the loss "- 10 * log10(MSE)". As TF wants to minimize an objective
    function, we implement -psnr.

    Parameters
    ----------
    ground_truth : TYPE
        sharp image, clean image
    prediction : TYPE
        blurry image, image with noise

    Returns
    -------
    scalar
        negative psnr
    """

    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    with tf.variable_scope("psnr_loss"):
        return 10. * log10(tf.reduce_mean(tf.square(prediction - ground_truth)))


def sobel_filter(x):
    """Compute image gradient using Sobel-filter.

    Parameters
    ----------
    x : TYPE
        any tensor

    Returns
    -------
    TYPE
        pair of image-gradient [dx, dy]
    """
    filter_values = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x = tf.reshape(filter_values, [3, 3, 1, 1])
    sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])

    dx = tf.nn.conv2d(x, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    dy = tf.nn.conv2d(x, sobel_y, strides=[1, 1, 1, 1], padding='SAME')

    return dx, dy


@contextmanager
def GuidedRelu():  # noqa
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import gen_nn_ops

    @ops.RegisterGradient("GuidedRelu")
    def _GuidedReluGrad(op, grad):  # noqa
        # guided backprop
        return tf.where(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        yield
