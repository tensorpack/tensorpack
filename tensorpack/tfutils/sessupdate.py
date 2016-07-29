#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sessupdate.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import tensorflow as tf

__all__ = ['SessionUpdate']

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
