# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np


def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
    """
    Args:
        logits: (N,C)
        label: (N,)
        topk(int): topk
    Returns:
        a float32 vector of length N with 0/1 values. 1 means incorrect
        prediction.
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

    Args:
        pred: of shape (b, ...). the predictions in [0,1].
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    z = batch_flatten(pred)
    y = tf.cast(batch_flatten(label), tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    eps = 1e-12
    loss_pos = -beta * tf.reduce_mean(y * tf.log(z + eps))
    loss_neg = (1. - beta) * tf.reduce_mean((1. - y) * tf.log(1. - z + eps))
    cost = tf.subtract(loss_pos, loss_neg, name=name)
    return cost


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    This function accepts logits rather than predictions, and is more numerically stable than
    :func:`class_balanced_cross_entropy`.
    """
    y = tf.cast(label, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)
    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
    cost = tf.reduce_mean(cost * (1 - beta), name=name)
    return cost


def print_stat(x, message=None):
    """ A simple print Op that might be easier to use than :meth:`tf.Print`.
        Use it like: ``x = print_stat(x, message='This is x')``.
    """
    if message is None:
        message = x.op.name
    return tf.Print(x, [tf.shape(x), tf.reduce_mean(x), x], summarize=20,
                    message=message, name='print_' + x.op.name)


def rms(x, name=None):
    """
    Returns:
        root mean square of tensor x.
    """
    if name is None:
        name = x.op.name + '/rms'
        with tf.name_scope(None):   # name already contains the scope
            return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)


def huber_loss(x, delta=1, name='huber_loss'):
    r"""
    Huber loss of x.

    .. math::

        y = \begin{cases} \frac{x^2}{2}, & |x| < \delta \\
        \delta |x| - \frac{\delta^2}{2}, & |x| \ge \delta
        \end{cases}

    Args:
        x: the difference vector.
        delta (float):

    Returns:
        a tensor of the same shape of x.
    """
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.where(abscost < delta,
                    sqrcost * 0.5,
                    abscost * delta - 0.5 * delta ** 2,
                    name=name)


def get_scalar_var(name, init_value, summary=False, trainable=False):
    """
    Get a scalar variable with certain initial value

    Args:
        name (str): name of the variable.
        init_value (float): initial value.
        summary (bool): whether to summary this variable.
        trainable (bool): trainable or not.
    Returns:
        tf.Variable: the variable
    """
    ret = tf.get_variable(name, shape=[],
                          initializer=tf.constant_initializer(init_value),
                          trainable=trainable)
    if summary:
        # this is recognized in callbacks.StatHolder
        tf.summary.scalar(name + '-summary', ret)
    return ret
