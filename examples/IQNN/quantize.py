#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dorefa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorpack.utils.argtools import memoized


@memoized
def get_quantize(bitW, bitA):
    """
    return the three quantization functions fw, fa, fg, for weights, activations and gradients respectively
    It's unsafe to call this function multiple times with different parameters
    """
    G = tf.get_default_graph()

    def quantize(x, k):
        x = tf.clip_by_value(x, -0.5, 0.5)
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round((x + 0.5) * n) / n - 0.5

    def fw(x):
        if bitW == 32:
            return x
        x = tf.tanh(x)
        scale = tf.reduce_mean(tf.abs(x)) * 2
        y = quantize(x / scale, bitW) * scale
        return y

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    return fw, fa

@memoized
def get_iter_quantize(bitW, bitA, num_iter):
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k - 1)
        with G.gradient_override_map({"Round": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        if bitW == 32:
            return x
        eps = 1e-32
        x = tf.tanh(x)
        x = tf.transpose(x, perm=[3, 2, 0, 1])
        w = tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))
        a2 = tf.expand_dims(tf.reduce_mean(w, axis=1), 1)
        a1 = tf.expand_dims(4 * tf.reduce_mean(tf.abs(w - a2), axis=1), 1)
        b = quantize(tf.clip_by_value((w - a2) / a1, -1, 1) * 0.5, bitW) * 2
        for i in range(num_iter):
            a1 = tf.expand_dims(tf.maximum(eps, tf.reduce_sum((w - a2) * b, axis=1)) / tf.maximum(eps, tf.reduce_sum(b * b, axis=1)), axis=1)
            a2 = tf.expand_dims(tf.reduce_mean(w - a1 * b, axis=1), axis=1)
            b = quantize(tf.clip_by_value((w - a2) / a1, -1, 1) * 0.5, bitW) * 2
        y = a1 * b + a2
        y = tf.reshape(y, tf.shape(x))
        y = tf.transpose(y, perm=[2, 3, 1, 0])
        return y

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    return fw, fa
