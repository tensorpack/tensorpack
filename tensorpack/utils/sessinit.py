#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: sessinit.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod
import tensorflow as tf

from . import logger
class SessionInit(object):
    @abstractmethod
    def init(self, sess):
        """ Method to initialize a session"""

class NewSession(SessionInit):
    def init(self, sess):
        sess.run(tf.initialize_all_variables())

class SaverRestore(SessionInit):
    def __init__(self, model_path):
        self.set_path(model_path)

    def init(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, self.path)
        logger.info(
            "Restore checkpoint from {}".format(ckpt.model_checkpoint_path))

    def set_path(self, model_path):
        self.path = model_path

class ParamRestore(SessionInit):
    def __init__(self, param_dict):
        self.prms = param_dict

    def init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_dict = dict([v.name, v] for v in variables)
        for name, value in self.prms.iteritems():
            try:
                var = var_dict[name]
            except (ValueError, KeyError):
                logger.warn("Param {} not found in this graph".format(name))
                continue
            logger.info("Restoring param {}".format(name))
            sess.run(var.assign(value))
