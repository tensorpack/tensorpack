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

@contextmanager
def timed_operation(msg, log_start=False):
    if log_start:
        logger.info('start {} ...'.format(msg))
    start = time.time()
    yield
    logger.info('finished {}, time={:.2f}sec.'.format(
        msg, time.time() - start))

@contextmanager
def create_test_graph():
    G = tf.get_default_graph()
    input_vars_train = G.get_collection(INPUT_VARS_KEY)
    forward_func = G.get_collection(FORWARD_FUNC_KEY)[0]
    with tf.Graph().as_default() as Gtest:
        input_vars = []
        for v in input_vars_train:
            name = v.name
            assert name.endswith(':0'), "I think placeholder variable should all ends with ':0'"
            name = name[:-2]
            input_vars.append(tf.placeholder(
                v.dtype, shape=v.get_shape(), name=name
            ))
        for v in input_vars:
            Gtest.add_to_collection(INPUT_VARS_KEY, v)
        output_vars, cost = forward_func(input_vars, is_training=False)
        for v in output_vars:
            Gtest.add_to_collection(OUTPUT_VARS_KEY, v)
        yield Gtest

@contextmanager
def create_test_session():
    with create_test_graph():
        with tf.Session() as sess:
            yield sess
