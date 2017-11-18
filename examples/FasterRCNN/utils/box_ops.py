#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: box_ops.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils import get_default_sess_config
from tensorpack.utils.argtools import memoized

"""
This file is modified from
https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py
"""


@under_name_scope()
def area(boxes):
    """
    Args:
      boxes: nx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


@memoized
def get_iou_callable():
    """
    Get a pairwise box iou callable.
    """
    # We don't want the dataflow process to touch CUDA
    # Data needs tensorflow. As a result, the training cannot run on GPUs with
    # EXCLUSIVE_PROCESS mode, unless you disable multiprocessing prefetch.
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        A = tf.placeholder(tf.float32, shape=[None, 4])
        B = tf.placeholder(tf.float32, shape=[None, 4])
        iou = pairwise_iou(A, B)
        sess = tf.Session(config=get_default_sess_config())
        return sess.make_callable(iou, [A, B])
