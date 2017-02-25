#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sesscreate.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

__all__ = ['NewSessionCreator', 'ReuseSessionCreator']


class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target='', graph=None, config=None):
        """
        Args:
            target, graph, config: same as :meth:`Session.__init__()`.
        """
        self.target = target
        self.config = config
        self.graph = graph

    def create_session(self):
        return tf.Session(target=self.target, graph=self.graph, config=self.config)


class ReuseSessionCreator(tf.train.SessionCreator):
    def __init__(self, sess):
        """
        Args:
            sess (tf.Session): the session to reuse
        """
        self.sess = sess

    def create_session(self):
        return self.sess
