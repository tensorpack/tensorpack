# -*- coding: UTF-8 -*-
# File: symbolic_functions.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
import numpy as np


# this function exists for backwards-compatibilty
def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
    """
    Args:
        logits: shape [B,C].
        label: shape [B].
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
    cost = tf.reduce_mean(cost * (1 - beta))
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


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
    Get a scalar float variable with certain initial value

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


def psnr(prediction, ground_truth, name='psnr'):
    """`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

    .. math::

        PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)

    This function assumes the maximum possible value of the signal is 1,
    therefore the PSNR is simply :math:`- 10 \cdot \log_{10}(MSE)`.

    Args:
        prediction: a :class:`tf.Tensor` representing the prediction signal.
        ground_truth: another :class:`tf.Tensor` with the same shape.

    Returns:
        A scalar tensor representing the PSNR.
    """

    def log10(x):
        numerator = tf.log(x)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    return tf.multiply(log10(tf.reduce_mean(tf.square(prediction - ground_truth))),
                       -10., name=name)


@contextmanager
def guided_relu():
    """
    Returns:
        A context where the gradient of :meth:`tf.nn.relu` is replaced by
        guided back-propagation, as described in the paper:
        `Striving for Simplicity: The All Convolutional Net
        <https://arxiv.org/abs/1412.6806>`_
    """
    from tensorflow.python.ops import gen_nn_ops   # noqa

    @tf.RegisterGradient("GuidedReLU")
    def GuidedReluGrad(op, grad):
        return tf.where(0. < grad,
                        gen_nn_ops._relu_grad(grad, op.outputs[0]),
                        tf.zeros(grad.get_shape()))

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedReLU'}):
        yield


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


def contrastive_loss(left, right, y, margin, extra=False, scope="constrastive_loss"):
    r"""Loss for Siamese networks as described in the paper:
    `Learning a Similarity Metric Discriminatively, with Application to Face
    Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_ by Chopra et al.

    .. math::
        \frac{1}{2} [y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2], d = \Vert l - r \Vert_2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.
        margin (float): horizon for negative examples (y==0).
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: constrastive_loss (averaged over the batch), (and optionally average_pos_dist, average_neg_dist)
    """
    with tf.name_scope(scope):
        y = tf.cast(y, tf.float32)

        delta = tf.reduce_sum(tf.square(left - right), 1)
        delta_sqrt = tf.sqrt(delta + 1e-10)

        match_loss = delta
        missmatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

        loss = tf.reduce_mean(0.5 * (y * match_loss + (1 - y) * missmatch_loss))

        if extra:
            num_pos = tf.count_nonzero(y)
            num_neg = tf.count_nonzero(1 - y)
            pos_dist = tf.where(tf.equal(num_pos, 0), 0.,
                                tf.reduce_sum(y * delta_sqrt) / tf.cast(num_pos, tf.float32),
                                name="pos-dist")
            neg_dist = tf.where(tf.equal(num_neg, 0), 0.,
                                tf.reduce_sum((1 - y) * delta_sqrt) / tf.cast(num_neg, tf.float32),
                                name="neg-dist")
            return loss, pos_dist, neg_dist
        else:
            return loss


def cosine_loss(left, right, y, scope="cosine_loss"):
    r"""Loss for Siamese networks (cosine version).
    Same as :func:`contrastive_loss` but with different similarity measurement.

    .. math::
        [\frac{l \cdot r}{\lVert l\rVert \lVert r\rVert} - (2y-1)]^2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.

    Returns:
        tf.Tensor: cosine-loss as a scalar tensor.
    """

    def l2_norm(t, eps=1e-12):
        """
        Returns:
            tf.Tensor: norm of 2D input tensor on axis 1
        """
        with tf.name_scope("l2_norm"):
            return tf.sqrt(tf.reduce_sum(tf.square(t), 1) + eps)

    with tf.name_scope(scope):
        y = 2 * tf.cast(y, tf.float32) - 1
        pred = tf.reduce_sum(left * right, 1) / (l2_norm(left) * l2_norm(right) + 1e-10)

        return tf.nn.l2_loss(y - pred) / tf.cast(tf.shape(left)[0], tf.float32)


def triplet_loss(anchor, positive, negative, margin, extra=False, scope="triplet_loss"):
    r"""Loss for Triplet networks as described in the paper:
    `FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_
    by Schroff et al.

    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).

    .. math::
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin (float): horizon for negative examples
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def soft_triplet_loss(anchor, positive, negative, extra=True, scope="soft_triplet_loss"):
    """Loss for triplet networks as described in the paper:
    `Deep Metric Learning using Triplet Network
    <https://arxiv.org/abs/1412.6622>`_ by Hoffer et al.

    It is a softmax loss using :math:`(anchor-positive)^2` and
    :math:`(anchor-negative)^2` as logits.

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), 1) + eps)
        d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), 1) + eps)

        logits = tf.stack([d_pos, d_neg], axis=1)
        ones = tf.ones_like(tf.squeeze(d_pos), dtype="int32")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ones))

        if extra:
            pos_dist = tf.reduce_mean(d_pos, name='pos-dist')
            neg_dist = tf.reduce_mean(d_neg, name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


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
