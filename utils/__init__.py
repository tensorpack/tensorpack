# !/usr/bin/env python2
#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: Yuxin Wu <ppwwyyxx@gmail.com>

from pkgutil import walk_packages
import os
import time
import sys
from contextlib import contextmanager
import logger
import tensorflow as tf

def global_import(name):
    p = __import__(name, globals(), locals())
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        globals()[k] = p.__dict__[k]

global_import('naming')
global_import('callback')
global_import('validation_callback')


@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('finished {}, time={:.2f}sec.'.format(
        msg, time.time() - start))


def summary_model():
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

