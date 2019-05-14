#!/usr/bin/env python

import tensorflow as tf


def backport_tensor_spec():
    if hasattr(tf, 'TensorSpec'):
        return tf.TensorSpec
    try:
        # available since 1.7
        from tensorflow.python.framework.tensor_spec import TensorSpec
    except ImportError:
        pass
    else:
        tf.TensorSpec = TensorSpec
        return TensorSpec

    from .tensor_spec import TensorSpec
    tf.TensorSpec = TensorSpec
    return TensorSpec


def is_tfv2():
    try:
        from tensorflow.python import tf2
        return tf2.enabled()
    except Exception:
        return False


if is_tfv2():
    tfv1 = tf.compat.v1
    if not hasattr(tf, 'layers'):
        # promised at https://github.com/tensorflow/community/pull/24#issuecomment-440453886
        tf.layers = tf.keras.layers
else:
    try:
        tfv1 = tf.compat.v1  # this will silent some warnings
    except AttributeError:
        tfv1 = tf
