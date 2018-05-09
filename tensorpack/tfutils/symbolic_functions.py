# -*- coding: utf-8 -*-
# File: symbolic_functions.py


import tensorflow as tf
import numpy as np

from ..utils.develop import deprecated

# __all__ = ['get_scalar_var']


# this function exists for backwards-compatibility
def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
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


def print_stat(x, message=None):
    """ A simple print Op that might be easier to use than :meth:`tf.Print`.
        Use it like: ``x = print_stat(x, message='This is x')``.
    """
    if message is None:
        message = x.op.name
    lst = [tf.shape(x), tf.reduce_mean(x)]
    if x.dtype.is_floating:
        lst.append(rms(x))
    return tf.Print(x, lst + [x], summarize=20,
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


@deprecated("Please use tf.losses.huber_loss instead!")
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
    with tf.name_scope('huber_loss'):
        sqrcost = tf.square(x)
        abscost = tf.abs(x)

        cond = abscost < delta
        l2 = sqrcost * 0.5
        l1 = abscost * delta - 0.5 * delta ** 2
    return tf.where(cond, l2, l1, name=name)


# TODO deprecate this in the future
# doesn't hurt to keep it here for now
def get_scalar_var(name, init_value, summary=False, trainable=False):
    """
    Get a scalar float variable with certain initial value.
    You can just call `tf.get_variable(name, initializer=init_value, trainable=False)` instead.

    Args:
        name (str): name of the variable.
        init_value (float): initial value.
        summary (bool): whether to summary this variable.
        trainable (bool): trainable or not.
    Returns:
        tf.Variable: the variable
    """
    ret = tf.get_variable(name, initializer=float(init_value),
                          trainable=trainable)
    if summary:
        # this is recognized in callbacks.StatHolder
        tf.summary.scalar(name + '-summary', ret)
    return ret


@deprecated("Please implement it by yourself.", "2018-04-28")
def psnr(prediction, ground_truth, maxp=None, name='psnr'):
    """`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    .. math::

        PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)

    Args:
        prediction: a :class:`tf.Tensor` representing the prediction signal.
        ground_truth: another :class:`tf.Tensor` with the same shape.
        maxp: maximum possible pixel value of the image (255 in in 8bit images)

    Returns:
        A scalar tensor representing the PSNR.
    """

    maxp = float(maxp)

    def log10(x):
        with tf.name_scope("log10"):
            numerator = tf.log(x)
            denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

    mse = tf.reduce_mean(tf.square(prediction - ground_truth))
    if maxp is None:
        psnr = tf.multiply(log10(mse), -10., name=name)
    else:
        psnr = tf.multiply(log10(mse), -10.)
        psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)

    return psnr


@deprecated("Please implement it by yourself.", "2018-04-28")
def saliency_map(output, input, name="saliency_map"):
    """
    Produce a saliency map as described in the paper:
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/abs/1312.6034>`_.
    The saliency map is the gradient of the max element in output w.r.t input.

    Returns:
        tf.Tensor: the saliency map. Has the same shape as input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.gradients(max_outp, input)[:][0]
    saliency_op = tf.identity(saliency_op, name=name)
    return saliency_op


@deprecated("Please implement it by yourself.", "2018-04-28")
def shapeless_placeholder(x, axis, name):
    """
    Make the static shape of a tensor less specific.

    If you want to feed to a tensor, the shape of the feed value must match
    the tensor's static shape. This function creates a placeholder which
    defaults to x if not fed, but has a less specific static shape than x.
    See also `tensorflow#5680
    <https://github.com/tensorflow/tensorflow/issues/5680>`_.

    Args:
        x: a tensor
        axis(int or list of ints): these axes of ``x.get_shape()`` will become
            None in the output.
        name(str): name of the output tensor

    Returns:
        a tensor equal to x, but shape information is partially cleared.
    """
    shp = x.get_shape().as_list()
    if not isinstance(axis, list):
        axis = [axis]
    for a in axis:
        if shp[a] is None:
            raise ValueError("Axis {} of shape {} is already unknown!".format(a, shp))
        shp[a] = None
    x = tf.placeholder_with_default(x, shape=shp, name=name)
    return x
