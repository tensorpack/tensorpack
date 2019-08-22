# -*- coding: utf-8 -*-
# File: symbolic_functions.py


import tensorflow as tf

from ..compat import tfv1

__all__ = ['print_stat', 'rms']


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


# for internal use only
def rms(x, name=None):
    """
    Returns:
        root mean square of tensor x.
    """
    if name is None:
        name = x.op.name + '/rms'
        with tfv1.name_scope(None):   # name already contains the scope
            return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)
    return tf.sqrt(tf.reduce_mean(tf.square(x)), name=name)


# don't hurt to leave it here
def psnr(prediction, ground_truth, maxp=None, name='psnr'):
    """`Peak Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    .. math::

        PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)

    Args:
        prediction: a :class:`tf.Tensor` representing the prediction signal.
        ground_truth: another :class:`tf.Tensor` with the same shape.
        maxp: maximum possible pixel value of the image (255 in in 8bit images)

    Returns:
        A scalar tensor representing the PSNR
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
