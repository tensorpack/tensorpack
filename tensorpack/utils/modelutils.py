#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: modelutils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf

from . import logger

def describe_model():
    """ describe the current model parameters"""
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    msg = [""]
    total = 0
    for v in train_vars:
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        msg.append("{}: shape={}, dim={}".format(
            v.name, shape.as_list(), ele))
    msg.append("Total dim={}".format(total))
    logger.info("Model Params: {}".format('\n'.join(msg)))


def get_shape_str(tensors):
    """ return the shape string for a tensor or a list of tensors"""
    if isinstance(tensors, (list, tuple)):
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        shape_str = str(tensors.get_shape().as_list())
    return shape_str

