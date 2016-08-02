#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varmanip.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import tensorflow as tf
import numpy as np

__all__ = ['SessionUpdate', 'dump_session_params', 'dump_chkpt_vars']

class SessionUpdate(object):
    """ Update the variables in a session """
    def __init__(self, sess, vars_to_update):
        """
        :param vars_to_update: a collection of variables to update
        """
        self.sess = sess
        self.assign_ops = {}
        for v in vars_to_update:
            p = tf.placeholder(v.dtype, shape=v.get_shape())
            self.assign_ops[v.name] = (p, v.assign(p))

    def update(self, prms):
        """
        :param prms: dict of {variable name: value}
        Any name in prms must be in the graph and in vars_to_update.
        """
        for name, value in six.iteritems(prms):
            p, op = self.assign_ops[name]
            varshape = tuple(p.get_shape().as_list())
            if varshape != value.shape:
                # TODO only allow reshape when shape different by empty axis
                assert np.prod(varshape) == np.prod(value.shape), \
                        "{}: {}!={}".format(name, varshape, value.shape)
                logger.warn("Param {} is reshaped during assigning".format(name))
                value = value.reshape(varshape)
            self.sess.run(op, feed_dict={p: value})

def dump_session_params(path):
    """ Dump value of all trainable + to_save variables to a dict and save to `path` as
    npy format, loadable by ParamRestore
    """
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var.extend(tf.get_collection(EXTRA_SAVE_VARS_KEY))
    result = {}
    for v in var:
        name = v.name.replace(":0", "")
        result[name] = v.eval()
    logger.info("Variables to save to {}:".format(path))
    logger.info(str(result.keys()))
    np.save(path, result)

def dump_chkpt_vars(model_path, output):
    """ Dump all variables from a checkpoint """
    reader = tf.train.NewCheckpointReader(model_path)
    var_names = reader.get_variable_to_shape_map().keys()
    result = {}
    for n in var_names:
        result[n] = reader.get_tensor(n)
    logger.info("Variables to save to {}:".format(output))
    logger.info(str(result.keys()))
    np.save(output, result)
